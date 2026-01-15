#!/usr/bin/env python3
import os
import json
import yaml
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import wandb

from model import TSCLTrainer, BaselineTrainer, PerplexityRewardTrainer, TSCLOutput


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class MathReasoningDataset(Dataset):

    def __init__(
        self,
        dataset_name: str = "gsm8k",
        split: str = "train",
        tokenizer: AutoTokenizer = None,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if dataset_name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split=split)
        elif dataset_name == "math":
            ds = load_dataset("competition_math", split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.data = list(ds)
        if max_samples:
            self.data = self.data[:max_samples]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        if "question" in item:
            question = item["question"]
            answer = item["answer"]
        else:
            question = item["problem"]
            answer = item["solution"]

        return {
            "question": question,
            "answer": answer,
            "idx": idx,
        }


def extract_number(text: str) -> Optional[float]:
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def verify_answer(generated: str, ground_truth: str) -> bool:
    gen_num = extract_number(generated)
    gt_num = extract_number(ground_truth)

    if gen_num is None or gt_num is None:
        return False

    return abs(gen_num - gt_num) < 1e-3


class TrajectoryGenerator:

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: torch.nn.Module,
        max_length: int = 512,
        num_trajectories: int = 8,
        device: str = "cuda",
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.num_trajectories = num_trajectories
        self.device = device

    def generate(
        self,
        questions: List[str],
        answers: List[str],
    ) -> Dict:
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_rewards = []
        all_texts = []
        all_query_hidden = []

        for q, a in zip(questions, answers):
            prompt = f"Question: {q}\n\nSolve this step by step.\n\nAnswer:"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length // 2,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                query_outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )
                query_hidden = query_outputs.hidden_states[-1][:, -1, :]

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length // 2,
                    num_return_sequences=self.num_trajectories,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for i, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)

                is_correct = verify_answer(generated_text, a)

                full_inputs = self.tokenizer(
                    generated_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                )

                all_input_ids.append(full_inputs["input_ids"].to(self.device))
                all_attention_masks.append(full_inputs["attention_mask"].to(self.device))
                all_labels.append(1 if is_correct else 0)
                all_rewards.append(1.0 if is_correct else -1.0)
                all_texts.append(generated_text)
                all_query_hidden.append(query_hidden)

        return {
            "input_ids": torch.cat(all_input_ids, dim=0),
            "attention_mask": torch.cat(all_attention_masks, dim=0),
            "correctness": torch.tensor(all_labels, dtype=torch.long, device=self.device),
            "rewards": torch.tensor(all_rewards, dtype=torch.float, device=self.device),
            "texts": all_texts,
            "query_hidden": torch.cat(all_query_hidden, dim=0),
        }


def compute_pass_at_k(
    correct_per_problem: List[int],
    n: int,
    k: int,
) -> float:
    def pass_at_k_single(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.mean([pass_at_k_single(n, c, k) for c in correct_per_problem])


def train_epoch_tscl(
    trainer: TSCLTrainer,
    dataloader: DataLoader,
    generator: TrajectoryGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Dict,
    epoch: int,
    device: str = "cuda",
) -> Dict:
    trainer.model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_intrinsic_loss = 0.0
    strategy_surprises = []
    success_surprises = []
    diversity_scores = []
    accuracy_per_problem = []

    compute_diversity_every = config["evaluation"].get("eval_steps", 25)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        questions = batch["question"]
        answers = batch["answer"]

        trajectory_data = generator.generate(questions, answers)

        compute_diversity = (step % compute_diversity_every == 0)

        output: TSCLOutput = trainer.forward(
            input_ids=trajectory_data["input_ids"],
            attention_mask=trajectory_data["attention_mask"],
            generated_texts=trajectory_data["texts"],
            correctness=trajectory_data["correctness"],
            rewards=trajectory_data["rewards"],
            query_hidden_states=trajectory_data["query_hidden"],
            compute_diversity=compute_diversity,
        )

        optimizer.zero_grad()
        output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.get_trainable_parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += output.total_loss.item()
        total_policy_loss += output.policy_loss.item()
        total_intrinsic_loss += output.intrinsic_loss.item() if isinstance(output.intrinsic_loss, torch.Tensor) else output.intrinsic_loss
        strategy_surprises.append(output.strategy_surprise)
        success_surprises.append(output.success_surprise)

        correct = trajectory_data["correctness"].cpu().numpy()
        n_traj = config["training"]["num_trajectories_per_problem"]
        for i in range(0, len(correct), n_traj):
            accuracy_per_problem.append(correct[i:i+n_traj].sum())

        if output.diversity_metrics:
            diversity_scores.append(output.diversity_metrics["strategy_cluster_entropy"])

        pbar.set_postfix({
            "loss": output.total_loss.item(),
            "ss": output.strategy_surprise,
            "sus": output.success_surprise,
        })

        if wandb.run:
            log_dict = {
                "train/loss": output.total_loss.item(),
                "train/policy_loss": output.policy_loss.item(),
                "train/intrinsic_loss": output.intrinsic_loss.item() if isinstance(output.intrinsic_loss, torch.Tensor) else output.intrinsic_loss,
                "train/strategy_surprise": output.strategy_surprise,
                "train/success_surprise": output.success_surprise,
                "train/pred_success": output.mean_pred_success,
                "train/lr": scheduler.get_last_lr()[0],
            }
            if output.diversity_metrics:
                log_dict["train/strategy_cluster_entropy"] = output.diversity_metrics["strategy_cluster_entropy"]
                log_dict["train/correct_diversity"] = output.diversity_metrics["correct_diversity"]
            wandb.log(log_dict)

    num_steps = len(dataloader)
    n_traj = config["training"]["num_trajectories_per_problem"]
    k_values = config["evaluation"].get("k_values", [1, 5, 10])

    metrics = {
        "loss": total_loss / num_steps,
        "policy_loss": total_policy_loss / num_steps,
        "intrinsic_loss": total_intrinsic_loss / num_steps,
        "strategy_surprise_mean": np.mean(strategy_surprises),
        "strategy_surprise_std": np.std(strategy_surprises),
        "success_surprise_mean": np.mean(success_surprises),
        "success_surprise_std": np.std(success_surprises),
        "strategy_cluster_entropy": np.mean(diversity_scores) if diversity_scores else 0.0,
    }

    for k in k_values:
        if k <= n_traj:
            metrics[f"pass@{k}"] = compute_pass_at_k(accuracy_per_problem, n_traj, k)

    return metrics


def train_epoch_baseline(
    trainer: BaselineTrainer,
    dataloader: DataLoader,
    generator: TrajectoryGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Dict,
    epoch: int,
    device: str = "cuda",
) -> Dict:
    trainer.model.train()

    total_loss = 0.0
    accuracy_per_problem = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Baseline]")
    for step, batch in enumerate(pbar):
        questions = batch["question"]
        answers = batch["answer"]

        trajectory_data = generator.generate(questions, answers)

        loss, metrics = trainer.forward(
            input_ids=trajectory_data["input_ids"],
            attention_mask=trajectory_data["attention_mask"],
            rewards=trajectory_data["rewards"],
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.get_trainable_parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        correct = trajectory_data["correctness"].cpu().numpy()
        n_traj = config["training"]["num_trajectories_per_problem"]
        for i in range(0, len(correct), n_traj):
            accuracy_per_problem.append(correct[i:i+n_traj].sum())

        pbar.set_postfix({"loss": loss.item()})

        if wandb.run:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
            })

    num_steps = len(dataloader)
    n_traj = config["training"]["num_trajectories_per_problem"]
    k_values = config["evaluation"].get("k_values", [1, 5, 10])

    result = {
        "loss": total_loss / num_steps,
    }

    for k in k_values:
        if k <= n_traj:
            result[f"pass@{k}"] = compute_pass_at_k(accuracy_per_problem, n_traj, k)

    return result


def train_epoch_perplexity(
    trainer: PerplexityRewardTrainer,
    dataloader: DataLoader,
    generator: TrajectoryGenerator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Dict,
    epoch: int,
    device: str = "cuda",
) -> Dict:
    trainer.model.train()

    total_loss = 0.0
    perplexity_rewards = []
    accuracy_per_problem = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Perplexity]")
    for step, batch in enumerate(pbar):
        questions = batch["question"]
        answers = batch["answer"]

        trajectory_data = generator.generate(questions, answers)

        loss, metrics = trainer.forward(
            input_ids=trajectory_data["input_ids"],
            attention_mask=trajectory_data["attention_mask"],
            rewards=trajectory_data["rewards"],
            correctness=trajectory_data["correctness"],
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.get_trainable_parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        perplexity_rewards.append(metrics.get("perplexity_reward_mean", 0.0))

        correct = trajectory_data["correctness"].cpu().numpy()
        n_traj = config["training"]["num_trajectories_per_problem"]
        for i in range(0, len(correct), n_traj):
            accuracy_per_problem.append(correct[i:i+n_traj].sum())

        pbar.set_postfix({"loss": loss.item()})

        if wandb.run:
            wandb.log({
                "train/loss": loss.item(),
                "train/perplexity_reward": metrics.get("perplexity_reward_mean", 0.0),
                "train/lr": scheduler.get_last_lr()[0],
            })

    num_steps = len(dataloader)
    n_traj = config["training"]["num_trajectories_per_problem"]
    k_values = config["evaluation"].get("k_values", [1, 5, 10])

    result = {
        "loss": total_loss / num_steps,
        "perplexity_reward_mean": np.mean(perplexity_rewards),
    }

    for k in k_values:
        if k <= n_traj:
            result[f"pass@{k}"] = compute_pass_at_k(accuracy_per_problem, n_traj, k)

    return result


def setup_model_and_tokenizer(config: Dict, device: str = "cuda"):
    print(f"Loading model: {config['model']['name']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["use_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if config["model"]["use_4bit"] else None

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if config["model"]["use_4bit"]:
        model = prepare_model_for_kbit_training(model)

    if config["model"]["use_lora"]:
        lora_config = LoraConfig(
            r=config["model"]["lora_r"],
            lora_alpha=config["model"]["lora_alpha"],
            lora_dropout=config["model"]["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    ref_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return model, ref_model, tokenizer


def main(args):
    config = load_config(args.config)
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = args.method
    run_name = f"{method_name}_seed{args.seed}_{timestamp}"
    output_dir = Path(config["output_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_wandb:
        wandb.init(
            project=config.get("wandb_project", "tscl-experiments"),
            name=run_name,
            config={**config, "seed": args.seed, "method": method_name},
        )

    model, ref_model, tokenizer = setup_model_and_tokenizer(config)
    device = str(model.device)
    hidden_dim = model.config.hidden_size

    if method_name == "tscl":
        trainer = TSCLTrainer(
            model=model,
            ref_model=ref_model,
            hidden_dim=hidden_dim,
            config=config,
            device=device,
        )
        train_fn = train_epoch_tscl
    elif method_name == "baseline":
        trainer = BaselineTrainer(
            model=model,
            ref_model=ref_model,
            config=config,
            device=device,
        )
        train_fn = train_epoch_baseline
    elif method_name == "perplexity":
        trainer = PerplexityRewardTrainer(
            model=model,
            ref_model=ref_model,
            config=config,
            device=device,
        )
        train_fn = train_epoch_perplexity
    else:
        raise ValueError(f"Unknown method: {method_name}")

    train_dataset = MathReasoningDataset(
        dataset_name=args.dataset,
        split="train",
        tokenizer=tokenizer,
        max_length=config["training"]["max_length"],
        max_samples=args.max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    generator = TrajectoryGenerator(
        tokenizer=tokenizer,
        model=model,
        max_length=config["training"]["max_length"],
        num_trajectories=config["training"]["num_trajectories_per_problem"],
        device=device,
    )

    optimizer = torch.optim.AdamW(
        trainer.get_trainable_parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * config["training"]["num_epochs"]
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Starting {method_name} training for {config['training']['num_epochs']} epochs")

    all_metrics = []
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        if method_name == "tscl":
            metrics = train_fn(
                trainer=trainer,
                dataloader=train_loader,
                generator=generator,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                device=device,
            )
        else:
            metrics = train_fn(
                trainer=trainer,
                dataloader=train_loader,
                generator=generator,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                device=device,
            )

        print(f"Epoch {epoch}: {metrics}")
        all_metrics.append({"epoch": epoch, **metrics})

        model.save_pretrained(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "model")

        if method_name == "tscl":
            torch.save({
                "sph": trainer.sph.state_dict(),
                "pse_projector": trainer.pse.projector.state_dict(),
            }, output_dir / "tscl_heads.pt")

    results = {
        "config": config,
        "seed": args.seed,
        "dataset": args.dataset,
        "method": method_name,
        "metrics": all_metrics,
        "final_metrics": all_metrics[-1] if all_metrics else {},
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Training complete. Results saved to {output_dir}")

    if wandb.run:
        wandb.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--method", type=str, default="tscl",
                        choices=["tscl", "baseline", "perplexity"])
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    main(args)
