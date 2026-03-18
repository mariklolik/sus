#!/usr/bin/env python3
"""Train baseline exploration methods with TRL GRPOTrainer."""

import os
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

from train_grpo import correctness_reward, format_reward, prepare_gsm8k, load_config, set_seed
from baselines import build_baseline


AVAILABLE_METHODS = [
    "entropy_bonus",
    "rs_grpo",
    "progrpo_arm",
    "ceeh_difficulty",
    "ucb_exploration",
]


def train(config_path: str, method: str, seed: int = 42):
    if method not in AVAILABLE_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {AVAILABLE_METHODS}")

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

    baseline_reward = build_baseline(method)
    reward_funcs = [correctness_reward, format_reward, baseline_reward]

    run_name = f"grpo_{method}_seed{seed}"
    output_dir = os.path.join(config.get("output_dir", "outputs"), run_name)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=training_cfg.get("num_epochs", 1),
        max_steps=training_cfg.get("max_steps", -1),
        per_device_train_batch_size=training_cfg.get("batch_size", 8),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 5e-6),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
        max_completion_length=training_cfg.get("max_completion_length", 1024),
        num_generations=training_cfg.get("num_trajectories_per_problem", 8),
        beta=grpo_cfg.get("kl_coef", 0.001),
        use_vllm=training_cfg.get("use_vllm", True),
        logging_steps=1,
        report_to="wandb" if os.environ.get("WANDB_MODE") != "disabled" else "none",
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        seed=seed,
    )

    wandb_project = config.get("wandb_project", "sus-paper")
    os.environ.setdefault("WANDB_PROJECT", wandb_project)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
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
    print(f"Training complete ({method}). Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--method", type=str, required=True, choices=AVAILABLE_METHODS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(config_path=args.config, method=args.method, seed=args.seed)
