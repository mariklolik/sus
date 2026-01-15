#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    {"method": "baseline", "config": "config_baseline.yaml", "name": "baseline"},
    {"method": "tscl", "config": "config.yaml", "name": "tscl_full"},
    {"method": "perplexity", "config": "config.yaml", "name": "perplexity"},
    {"method": "tscl", "config": "config_ablation_no_ss.yaml", "name": "tscl_no_ss"},
    {"method": "tscl", "config": "config_ablation_no_sus.yaml", "name": "tscl_no_sus"},
]

SEEDS = [42]
MAX_SAMPLES = 30


def run_experiment(method: str, config: str, seed: int, name: str):
    cmd = [
        sys.executable,
        "train.py",
        "--config", config,
        "--seed", str(seed),
        "--method", method,
        "--max_samples", str(MAX_SAMPLES),
        "--no_wandb",
    ]

    print(f"\n{'='*60}")
    print(f"Running: {name} seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    start_time = time.time()
    results = []

    for exp in EXPERIMENTS:
        for seed in SEEDS:
            try:
                ret = run_experiment(
                    method=exp["method"],
                    config=exp["config"],
                    seed=seed,
                    name=exp["name"],
                )
                results.append({
                    "name": exp["name"],
                    "seed": seed,
                    "success": ret == 0,
                })
            except Exception as e:
                print(f"Error in {exp['name']} seed={seed}: {e}")
                results.append({
                    "name": exp["name"],
                    "seed": seed,
                    "success": False,
                    "error": str(e),
                })

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All experiments completed in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['name']} seed={r['seed']}")


if __name__ == "__main__":
    main()
