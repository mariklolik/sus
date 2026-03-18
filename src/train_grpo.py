#!/usr/bin/env python3
"""Baseline GRPO training on GSM8K using TRL's GRPOTrainer."""

import os
import re
import yaml
import argparse
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve the given math problem step by step. "
    "Show your reasoning clearly, then provide the final numerical answer "
    "inside <answer> tags, like: <answer>42</answer>"
)


def make_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def prepare_gsm8k(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
):
    ds = load_dataset("gsm8k", "main", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_row(example):
        return {
            "prompt": make_prompt(example["question"], tokenizer),
            "ground_truth": example["answer"],
        }

    return ds.map(format_row, remove_columns=ds.column_names)


def extract_gsm8k_gt(ground_truth: str) -> Optional[str]:
    match = re.search(r"####\s*([\-\d,\.]+)", ground_truth)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def extract_generated(text: str) -> Optional[str]:
    match = re.search(r"<answer>\s*([\-\d,\.]+)\s*</answer>", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"\\boxed\{([\-\d,\.]+)\}", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None


def _numbers_match(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-3
    except (ValueError, TypeError):
        return False


def _completion_to_text(completion) -> str:
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    elif isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def correctness_reward(
    prompts: List[str],
    completions: List[str],
    ground_truth: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    if ground_truth is None:
        return [0.0] * len(completions)
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = _completion_to_text(completion)
        gt_num = extract_gsm8k_gt(gt)
        gen_num = extract_generated(text)
        if gt_num is not None and gen_num is not None and _numbers_match(gen_num, gt_num):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        if "<answer>" in text and "</answer>" in text:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str, seed: int = 42):
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

    run_name = f"grpo_baseline_seed{seed}"
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
        reward_funcs=[correctness_reward, format_reward],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(config_path=args.config, seed=args.seed)
