#!/usr/bin/env python3
"""SuS (Strategy-aware Surprise) training with TRL GRPOTrainer."""

import os
import argparse

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from train_grpo import load_config, prepare_gsm8k, set_seed
from sus_reward import SuSReward, SuSConfig


def train(config_path: str, ss_bonus: float = 0.2, err_bonus: float = 0.0,
          seed: int = 42, random_pse: bool = False):
    config = load_config(config_path)
    set_seed(seed)

    model_name = config["model"]["name"]
    training_cfg = config["training"]
    grpo_cfg = config.get("grpo", {})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_samples = config.get("evaluation", {}).get("num_eval_samples", None)
    train_dataset = prepare_gsm8k(tokenizer, split="train")
    eval_dataset = prepare_gsm8k(tokenizer, split="test", max_samples=eval_samples)

    lora_config = LoraConfig(
        r=config["model"].get("lora_r", 64),
        lora_alpha=config["model"].get("lora_alpha", 128),
        lora_dropout=config["model"].get("lora_dropout", 0.05),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sus_config = SuSConfig(ss_bonus=ss_bonus, err_bonus=err_bonus,
                           random_pse=random_pse)
    reward_fn = SuSReward(sus_config, device=device)
    reward_fn.__name__ = "sus_reward"

    # Single combined reward function — correctness, format, and novelty
    # are all handled internally to preserve zero-variance batches.
    reward_funcs = [reward_fn]

    bonus_str = f"{ss_bonus:.1f}".replace(".", "")
    err_str = f"{err_bonus:.2f}".replace(".", "")
    rnd_str = "_randpse" if random_pse else ""
    run_name = f"grpo_sus_b{bonus_str}_e{err_str}{rnd_str}_seed{seed}"
    output_dir = os.path.join(config.get("output_dir", "outputs"), run_name)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=training_cfg.get("num_epochs", 1),
        max_steps=training_cfg.get("max_steps", 2000),
        per_device_train_batch_size=training_cfg.get("batch_size", 8),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
        max_completion_length=training_cfg.get("max_completion_length", 1024),
        num_generations=training_cfg.get("num_trajectories_per_problem", 8),
        beta=grpo_cfg.get("kl_coef", 0.001),
        use_vllm=training_cfg.get("use_vllm", False),
        logging_steps=1,
        report_to="wandb" if os.environ.get("WANDB_MODE") != "disabled" else "none",
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        seed=seed,
    )

    wandb_project = config.get("wandb_project", "sus-paper")
    os.environ.setdefault("WANDB_PROJECT", wandb_project)

    # In distributed mode (torchrun), don't use device_map — let Trainer handle placement
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        **({} if distributed else {"device_map": "auto"}),
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        reward_funcs=reward_funcs,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete (ss_bonus={ss_bonus}, err_bonus={err_bonus}). Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--ss_bonus", type=float, default=0.1)
    parser.add_argument("--err_bonus", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_pse", action="store_true",
                        help="Replace MiniLM with random unit vectors (ablation)")
    args = parser.parse_args()
    train(config_path=args.config, ss_bonus=args.ss_bonus, err_bonus=args.err_bonus,
          seed=args.seed, random_pse=args.random_pse)
