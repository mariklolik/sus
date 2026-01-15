#!/usr/bin/env python3
"""Training script for SuS experiments."""

import argparse
import yaml
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from sus import SuSReward


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    
    print(f"Loading model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if config["model"]["use_lora"]:
        lora_config = LoraConfig(
            r=config["model"]["lora_r"],
            lora_alpha=config["model"]["lora_alpha"],
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    if config["sus"]["enabled"]:
        sus_reward = SuSReward(
            input_dim=model.config.hidden_size,
            strategy_dim=config["sus"]["strategy_dim"],
            lambda_ss=config["sus"]["lambda_ss"],
            lambda_sus=config["sus"]["lambda_sus"],
        )
        print(f"SuS enabled: λ_SS={config['sus']['lambda_ss']}, λ_SuS={config['sus']['lambda_sus']}")
    
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    print("Training...")
    # Training loop implementation here
    # See full implementation in experiments/
    
    output_path = Path(args.output_dir) / f"sus_seed{args.seed}"
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path / "model")
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
