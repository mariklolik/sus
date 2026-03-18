#!/usr/bin/env python3
"""Evaluate trained checkpoints on GSM8K test set.

Loads LoRA checkpoints, generates n_samples completions per problem,
computes pass@1, pass@5, pass@8 with bootstrap confidence intervals.
"""

import os
import re
import json
import math
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from train_grpo import SYSTEM_PROMPT


def extract_gsm8k_answer(ground_truth: str) -> Optional[float]:
    match = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", ground_truth)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def extract_generated_answer(generated: str) -> Optional[float]:
    match = re.search(r"<answer>\s*([-+]?\d[\d,]*\.?\d*)\s*</answer>", generated)
    if match:
        return float(match.group(1).replace(",", ""))
    match = re.search(r"\\boxed\{([-+]?\d[\d,]*\.?\d*)\}", generated)
    if match:
        return float(match.group(1).replace(",", ""))
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", generated)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None


def verify_answer(generated: str, ground_truth: str) -> bool:
    gen_num = extract_generated_answer(generated)
    gt_num = extract_gsm8k_answer(ground_truth)
    if gen_num is None or gt_num is None:
        return False
    return abs(gen_num - gt_num) < 1e-3


def compute_pass_at_k(correct_per_problem: List[int], n: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    def _single(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        num = sum(math.log(n - c - i) for i in range(k))
        den = sum(math.log(n - i) for i in range(k))
        return 1.0 - math.exp(num - den)
    return float(np.mean([_single(n, c, k) for c in correct_per_problem]))


def bootstrap_ci(values: List[float], n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values, dtype=float)
    means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)])
    alpha = 1.0 - ci
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def format_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_method(
    method: str,
    output_dir: str,
    n_samples: int = 16,
    batch_size: int = 8,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Dict:
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(output_dir, f"grpo_{method}_seed{seed}")

    if not os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        print(f"  [SKIP] No adapter found at {checkpoint_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"Evaluating: {method} (seed={seed})")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"{'='*60}")

    t0 = time.time()

    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()

    ds = load_dataset("gsm8k", "main", split="test")
    total_problems = len(ds)
    actual_end = end_idx if end_idx is not None else total_problems
    if start_idx > 0 or actual_end < total_problems:
        ds = ds.select(range(start_idx, actual_end))
        print(f"  GSM8K test: problems [{start_idx}:{actual_end}] ({len(ds)} of {total_problems}), {n_samples} samples each")
    else:
        print(f"  GSM8K test: {len(ds)} problems, {n_samples} samples each")

    correct_per_problem = []

    for idx, item in enumerate(tqdm(ds, desc=f"  {method}")):
        question = item["question"]
        ground_truth = item["answer"]
        prompt = format_prompt(question, tokenizer)

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)

        num_correct = 0
        for batch_start in range(0, n_samples, batch_size):
            cur_bs = min(batch_size, n_samples - batch_start)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    num_return_sequences=cur_bs,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for output in outputs:
                gen_ids = output[inputs["input_ids"].shape[1]:]
                generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if verify_answer(generated, ground_truth):
                    num_correct += 1

        correct_per_problem.append(num_correct)

        if (idx + 1) % 100 == 0:
            running_pass1 = compute_pass_at_k(correct_per_problem, n_samples, 1)
            print(f"    [{idx+1}/{len(ds)}] running pass@1 = {running_pass1:.4f}")

    results = {"method": method, "seed": seed, "n_samples": n_samples}
    results["correct_per_problem"] = correct_per_problem  # Raw counts for shard merging
    for k in [1, 5, 8]:
        if k <= n_samples:
            results[f"pass@{k}"] = compute_pass_at_k(correct_per_problem, n_samples, k)

    pass1_per_problem = [c / n_samples for c in correct_per_problem]
    ci_lo, ci_hi = bootstrap_ci(pass1_per_problem)
    results["pass@1_ci_95"] = [ci_lo, ci_hi]
    results["pass@1_mean_per_problem"] = float(np.mean(pass1_per_problem))
    results["num_problems"] = len(correct_per_problem)
    results["total_correct"] = sum(correct_per_problem)
    results["eval_time_seconds"] = time.time() - t0

    del model, base_model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    parser.add_argument("--method_name", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start problem index (for parallel eval)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End problem index (for parallel eval)")
    args = parser.parse_args()

    default_methods = ["baseline", "entropy_bonus", "rs_grpo", "progrpo_arm"]

    if args.method_name and args.checkpoint_dir:
        methods = [args.method_name]
        custom_checkpoints = {args.method_name: args.checkpoint_dir}
    else:
        methods = args.methods if args.methods else default_methods
        custom_checkpoints = {}

    all_results = {}

    print(f"GSM8K Evaluation")
    print(f"Methods: {methods}")
    print(f"n_samples={args.n_samples}, batch_size={args.batch_size}, seed={args.seed}")
    print(f"{'='*60}\n")

    for method in methods:
        try:
            result = evaluate_method(
                method=method,
                output_dir=args.output_dir,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                seed=args.seed,
                checkpoint_dir=custom_checkpoints.get(method),
                start_idx=args.start_idx,
                end_idx=args.end_idx,
            )
            if result:
                all_results[method] = result
                print(f"\n  >>> {method}: pass@1={result.get('pass@1', 0):.4f}  "
                      f"pass@5={result.get('pass@5', 0):.4f}  "
                      f"pass@8={result.get('pass@8', 0):.4f}  "
                      f"CI95={result.get('pass@1_ci_95', 'N/A')}  "
                      f"({result.get('eval_time_seconds', 0):.0f}s)")

                shard_suffix = f"_s{args.start_idx}_e{args.end_idx}" if args.end_idx else ""
                per_method_path = os.path.join(args.output_dir, f"eval_result_{method}{shard_suffix}.json")
                with open(per_method_path, "w") as f:
                    json.dump({method: result}, f, indent=2)
                out_path = os.path.join(args.output_dir, "eval_results_final.json")
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"\n  [ERROR] {method}: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {"error": str(e)}

    print(f"\n\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Pass@1':>8} {'Pass@5':>8} {'Pass@8':>8} {'CI95 (lo-hi)':>18}")
    print(f"{'-'*70}")
    for method in methods:
        r = all_results.get(method, {})
        if "error" in r:
            print(f"{method:<20} {'ERROR':>8}")
            continue
        if not r:
            print(f"{method:<20} {'SKIP':>8}")
            continue
        ci = r.get("pass@1_ci_95", [0, 0])
        print(f"{method:<20} {r.get('pass@1', 0):>8.4f} {r.get('pass@5', 0):>8.4f} "
              f"{r.get('pass@8', 0):>8.4f} {ci[0]:>8.4f}-{ci[1]:.4f}")

    out_path = os.path.join(args.output_dir, "eval_results_final.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
