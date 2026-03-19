#!/usr/bin/env python3
"""Evaluate trained checkpoints on MATH-500 (hendrycks/competition_math test split).

Generates n_samples completions per problem, extracts \boxed{} answers,
computes pass@1 and pass@8 with bootstrap CIs.
"""

import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")
import re
import json
import math
import argparse
import time
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import SYSTEM_PROMPT from the training code
sys.path.insert(0, os.path.dirname(__file__))
from train_grpo import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Answer extraction & verification for MATH
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """Normalize a MATH answer string for comparison."""
    answer = answer.strip()
    # Remove \text{...}, \mathrm{...}, \textbf{...}
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\textbf\{([^}]*)\}', r'\1', answer)
    # Remove \left and \right
    answer = answer.replace('\\left', '').replace('\\right', '')
    # Remove spaces
    answer = answer.replace(' ', '')
    # Remove trailing period
    answer = answer.rstrip('.')
    # Normalize \frac{a}{b} -> a/b for simple cases
    answer = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', answer)
    # Remove \$ signs
    answer = answer.replace('\\$', '').replace('$', '')
    # Normalize \% -> %
    answer = answer.replace('\\%', '%')
    # dfrac -> frac
    answer = answer.replace('\\dfrac', '\\frac')
    answer = answer.replace('\\tfrac', '\\frac')
    return answer


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content, handling nested braces."""
    # Find all \boxed occurrences, take the last one
    idx = text.rfind('\\boxed{')
    if idx == -1:
        idx = text.rfind('\\boxed ')
        if idx == -1:
            return None
        # \boxed without braces - grab next token
        rest = text[idx + 7:].strip()
        match = re.match(r'([^\s,;.]+)', rest)
        return match.group(1) if match else None

    # Navigate brace-matched content
    start = idx + 7  # len('\\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1]
    return None


def extract_answer_tag(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def try_numeric_compare(a: str, b: str) -> Optional[bool]:
    """Try to compare two strings as numbers. Return None if not possible."""
    try:
        # Try direct float parse
        va = float(a.replace(',', ''))
        vb = float(b.replace(',', ''))
        return abs(va - vb) < 1e-6
    except (ValueError, OverflowError):
        pass

    # Try evaluating simple arithmetic expressions only (fractions, powers)
    # Only if string looks safe (no letters except e for exponent)
    safe_pattern = re.compile(r'^[\d\s\+\-\*/\(\)\.\,\^eE]+$')
    if safe_pattern.match(a) and safe_pattern.match(b):
        try:
            va = eval(a.replace('^', '**').replace(',', ''))
            vb = eval(b.replace('^', '**').replace(',', ''))
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                return abs(float(va) - float(vb)) < 1e-6
        except:
            pass

    return None


def verify_math_answer(generated: str, ground_truth_solution: str, ground_truth_answer: str = None) -> bool:
    """Verify if the generated answer matches the MATH ground truth.

    ground_truth_solution is the full solution text; the answer is in \\boxed{}.
    ground_truth_answer is the clean answer string (if available from dataset).
    """
    # Use the clean answer field if available, otherwise extract from solution
    if ground_truth_answer is not None:
        gt_answer = ground_truth_answer.strip()
    else:
        gt_answer = extract_boxed(ground_truth_solution)
        if gt_answer is None:
            return False

    # Try to extract generated answer: first from <answer> tags, then from \boxed{}
    gen_answer = extract_answer_tag(generated)
    if gen_answer is None:
        gen_answer = extract_boxed(generated)
    if gen_answer is None:
        # Last resort: look for the last number
        numbers = re.findall(r'[-+]?\d[\d,]*\.?\d*', generated)
        if numbers:
            gen_answer = numbers[-1].replace(',', '')
        else:
            return False

    # Normalize both
    norm_gen = normalize_answer(gen_answer)
    norm_gt = normalize_answer(gt_answer)

    # Exact string match after normalization
    if norm_gen == norm_gt:
        return True

    # Numeric comparison
    num_result = try_numeric_compare(norm_gen, norm_gt)
    if num_result is not None:
        return num_result

    # Skip sympy (too slow / can hang). String + numeric matching covers most cases.
    return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def format_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_math500():
    """Load MATH-500 dataset. Try multiple sources."""
    # Try HuggingFaceH4/MATH-500 first (curated 500-problem subset, loads fastest)
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        print(f"  Loaded HuggingFaceH4/MATH-500: {len(ds)} problems")
        print(f"  Columns: {ds.column_names}")
        return ds
    except Exception as e:
        print(f"  HuggingFaceH4/MATH-500 failed: {e}")

    # Try hendrycks/competition_math
    try:
        ds = load_dataset("hendrycks/competition_math", split="test")
        print(f"  Loaded hendrycks/competition_math test: {len(ds)} problems")
        if len(ds) > 500:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(ds), size=500, replace=False)
            indices = sorted(indices.tolist())
            ds = ds.select(indices)
            print(f"  Sampled 500 problems")
        return ds
    except Exception as e:
        print(f"  hendrycks/competition_math failed: {e}")

    raise RuntimeError("Could not load any MATH dataset")


def evaluate_checkpoint_math500(
    method_name: str,
    checkpoint_dir: str,
    n_samples: int = 8,
    batch_size: int = 8,
) -> Dict:
    """Evaluate a single checkpoint on MATH-500."""

    print(f"\n{'='*60}")
    print(f"MATH-500 Evaluation: {method_name}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"n_samples={n_samples}, batch_size={batch_size}")
    print(f"{'='*60}")

    t0 = time.time()

    # Load model
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

    # Load dataset
    ds = load_math500()

    # Figure out field names (different datasets use different column names)
    columns = ds.column_names
    if "problem" in columns:
        question_field = "problem"
    elif "question" in columns:
        question_field = "question"
    else:
        raise ValueError(f"Cannot find question field in columns: {columns}")

    # For MATH-500, we have both 'solution' (full text) and 'answer' (clean answer)
    solution_field = "solution" if "solution" in columns else None
    answer_field = "answer" if "answer" in columns else None
    if solution_field is None and answer_field is None:
        raise ValueError(f"Cannot find solution/answer field in columns: {columns}")

    print(f"  Using fields: question='{question_field}', solution='{solution_field}', answer='{answer_field}'")
    print(f"  Dataset columns: {columns}")
    # Show a sample answer to verify format
    if answer_field:
        print(f"  Sample answer field: {ds[0][answer_field][:200]}")
    if solution_field:
        print(f"  Sample solution (first 200 chars): {ds[0][solution_field][:200]}")

    correct_per_problem = []
    subject_results = {}  # Track by subject/type if available

    for idx, item in enumerate(ds):
        question = item[question_field]
        gt_solution = item[solution_field] if solution_field else ""
        gt_answer = item[answer_field] if answer_field else None
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
                    max_new_tokens=768,
                    num_return_sequences=cur_bs,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for output in outputs:
                gen_ids = output[inputs["input_ids"].shape[1]:]
                generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if verify_math_answer(generated, gt_solution, gt_answer):
                    num_correct += 1

        correct_per_problem.append(num_correct)

        # Track by subject if available
        if "type" in columns:
            subj = item["type"]
        elif "subject" in columns:
            subj = item["subject"]
        else:
            subj = "all"
        if subj not in subject_results:
            subject_results[subj] = []
        subject_results[subj].append(num_correct)

        if (idx + 1) % 10 == 0 or idx == 0:
            running_pass1 = compute_pass_at_k(correct_per_problem, n_samples, 1)
            running_pass8 = compute_pass_at_k(correct_per_problem, n_samples, 8)
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(ds) - idx - 1)
            print(f"    [{idx+1}/{len(ds)}] pass@1={running_pass1:.4f}  pass@8={running_pass8:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    # Compute final metrics
    results = {
        "method": method_name,
        "dataset": "MATH-500",
        "n_samples": n_samples,
        "n_problems": len(ds),
        "correct_per_problem": correct_per_problem,
    }

    for k in [1, 5, 8]:
        if k <= n_samples:
            results[f"pass@{k}"] = compute_pass_at_k(correct_per_problem, n_samples, k)

    pass1_per_problem = [c / n_samples for c in correct_per_problem]
    ci_lo, ci_hi = bootstrap_ci(pass1_per_problem)
    results["pass@1_ci_95"] = [ci_lo, ci_hi]
    results["pass@1_mean_per_problem"] = float(np.mean(pass1_per_problem))

    # Per-subject breakdown
    subject_pass1 = {}
    for subj, counts in subject_results.items():
        subject_pass1[subj] = compute_pass_at_k(counts, n_samples, 1)
    results["per_subject_pass@1"] = subject_pass1

    results["total_correct"] = sum(correct_per_problem)
    results["eval_time_seconds"] = time.time() - t0

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  MATH-500 Results for {method_name}:")
    print(f"  pass@1 = {results['pass@1']:.4f}  CI95=[{ci_lo:.4f}, {ci_hi:.4f}]")
    if 'pass@5' in results:
        print(f"  pass@5 = {results['pass@5']:.4f}")
    print(f"  pass@8 = {results['pass@8']:.4f}")
    print(f"  Total correct samples: {results['total_correct']}/{len(ds)*n_samples}")
    if len(subject_pass1) > 1:
        print(f"  Per-subject pass@1:")
        for subj in sorted(subject_pass1.keys()):
            print(f"    {subj}: {subject_pass1[subj]:.4f}")
    print(f"  Eval time: {results['eval_time_seconds']:.0f}s")
    print(f"  {'='*50}")

    del model, base_model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on MATH-500")
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    results = evaluate_checkpoint_math500(
        method_name=args.method_name,
        checkpoint_dir=args.checkpoint_dir,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
