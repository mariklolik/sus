#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from scipy import stats


def load_results(output_dir: str) -> Dict[str, List[Dict]]:
    results_by_method = defaultdict(list)
    output_path = Path(output_dir)

    for run_dir in output_path.iterdir():
        if not run_dir.is_dir():
            continue

        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file, "r") as f:
            data = json.load(f)

        method = data.get("method", "unknown")

        config = data.get("config", {})
        tscl_config = config.get("tscl", {})
        lambda_strategy = tscl_config.get("lambda_strategy", 0.5)
        lambda_success = tscl_config.get("lambda_success", 0.5)

        if method == "tscl":
            if lambda_strategy == 0.0:
                method = "tscl_no_ss"
            elif lambda_success == 0.0:
                method = "tscl_no_sus"
            else:
                method = "tscl_full"

        results_by_method[method].append(data)

    return dict(results_by_method)


def compute_statistics(values: List[float]) -> Dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def run_ttest(group1: List[float], group2: List[float]) -> Dict:
    if len(group1) < 2 or len(group2) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}

    t_stat, p_value = stats.ttest_ind(group1, group2)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def extract_final_metrics(results: List[Dict], metric_name: str) -> List[float]:
    values = []
    for r in results:
        final = r.get("final_metrics", {})
        if metric_name in final:
            values.append(final[metric_name])
    return values


def analyze_hypothesis_h1(results_by_method: Dict) -> Dict:
    tscl_pass10 = extract_final_metrics(results_by_method.get("tscl_full", []), "pass@10")
    baseline_pass10 = extract_final_metrics(results_by_method.get("baseline", []), "pass@10")

    if not tscl_pass10 or not baseline_pass10:
        return {
            "hypothesis": "H1: TSCL achieves higher pass@k diversity",
            "supported": False,
            "reason": "Insufficient data",
        }

    tscl_stats = compute_statistics(tscl_pass10)
    baseline_stats = compute_statistics(baseline_pass10)
    ttest = run_ttest(tscl_pass10, baseline_pass10)

    improvement = (tscl_stats["mean"] - baseline_stats["mean"]) / (baseline_stats["mean"] + 1e-8) * 100

    return {
        "hypothesis": "H1: TSCL achieves higher pass@k diversity (≥15% improvement)",
        "tscl_pass@10": tscl_stats,
        "baseline_pass@10": baseline_stats,
        "improvement_percent": improvement,
        "ttest": ttest,
        "supported": improvement >= 15 and ttest["significant"],
    }


def analyze_hypothesis_h2(results_by_method: Dict) -> Dict:
    tscl_results = results_by_method.get("tscl_full", [])

    high_ss_ratios = []
    for r in tscl_results:
        for m in r.get("metrics", []):
            emb = m.get("embedding_metrics", {}) if isinstance(m, dict) else {}
            if "high_ss_correct_ratio" in emb:
                high_ss_ratios.append(emb["high_ss_correct_ratio"])

    return {
        "hypothesis": "H2: High Strategy Surprise correlates with novel correct solutions",
        "high_ss_correct_ratio_stats": compute_statistics(high_ss_ratios),
        "supported": len(high_ss_ratios) > 0 and np.mean(high_ss_ratios) > 0.1,
    }


def analyze_hypothesis_h3(results_by_method: Dict) -> Dict:
    tscl_entropy = extract_final_metrics(results_by_method.get("tscl_full", []), "strategy_cluster_entropy")
    perplexity_entropy = extract_final_metrics(results_by_method.get("perplexity", []), "strategy_cluster_entropy")

    if not tscl_entropy:
        return {
            "hypothesis": "H3: TSCL outperforms CDE on Strategy Cluster Entropy",
            "supported": False,
            "reason": "No TSCL entropy data",
        }

    tscl_stats = compute_statistics(tscl_entropy)

    if not perplexity_entropy:
        return {
            "hypothesis": "H3: TSCL outperforms CDE on Strategy Cluster Entropy",
            "tscl_entropy": tscl_stats,
            "supported": True,
            "reason": "No perplexity baseline for comparison",
        }

    perplexity_stats = compute_statistics(perplexity_entropy)
    improvement = (tscl_stats["mean"] - perplexity_stats["mean"]) / (perplexity_stats["mean"] + 1e-8) * 100
    ttest = run_ttest(tscl_entropy, perplexity_entropy)

    return {
        "hypothesis": "H3: TSCL outperforms CDE on Strategy Cluster Entropy (≥20%)",
        "tscl_entropy": tscl_stats,
        "perplexity_entropy": perplexity_stats,
        "improvement_percent": improvement,
        "ttest": ttest,
        "supported": improvement >= 20 and ttest["significant"],
    }


def analyze_hypothesis_h4(results_by_method: Dict) -> Dict:
    tscl_full = extract_final_metrics(results_by_method.get("tscl_full", []), "strategy_cluster_entropy")
    tscl_no_ss = extract_final_metrics(results_by_method.get("tscl_no_ss", []), "strategy_cluster_entropy")
    tscl_no_sus = extract_final_metrics(results_by_method.get("tscl_no_sus", []), "strategy_cluster_entropy")

    full_stats = compute_statistics(tscl_full)
    no_ss_stats = compute_statistics(tscl_no_ss)
    no_sus_stats = compute_statistics(tscl_no_sus)

    drop_without_ss = (full_stats["mean"] - no_ss_stats["mean"]) / (full_stats["mean"] + 1e-8) * 100 if tscl_no_ss else 0
    drop_without_sus = (full_stats["mean"] - no_sus_stats["mean"]) / (full_stats["mean"] + 1e-8) * 100 if tscl_no_sus else 0

    return {
        "hypothesis": "H4: Combined reward outperforms either component alone (≥10% drop)",
        "tscl_full_entropy": full_stats,
        "tscl_no_ss_entropy": no_ss_stats,
        "tscl_no_sus_entropy": no_sus_stats,
        "drop_without_ss_percent": drop_without_ss,
        "drop_without_sus_percent": drop_without_sus,
        "supported": (drop_without_ss >= 10 or drop_without_sus >= 10),
    }


def analyze_hypothesis_h5(results_by_method: Dict) -> Dict:
    tscl_results = results_by_method.get("tscl_full", [])
    baseline_results = results_by_method.get("baseline", [])

    tscl_ss_means = []
    for r in tscl_results:
        final = r.get("final_metrics", {})
        if "success_surprise_mean" in final:
            tscl_ss_means.append(final["success_surprise_mean"])

    return {
        "hypothesis": "H5: Success Surprise rewards 'surprising wins'",
        "success_surprise_stats": compute_statistics(tscl_ss_means),
        "note": "OOD evaluation requires separate test set",
        "supported": len(tscl_ss_means) > 0 and np.mean(tscl_ss_means) > 0.1,
    }


def analyze_hypothesis_h6(results_by_method: Dict) -> Dict:
    tscl_acc = extract_final_metrics(results_by_method.get("tscl_full", []), "pass@1")
    baseline_acc = extract_final_metrics(results_by_method.get("baseline", []), "pass@1")

    tscl_stats = compute_statistics(tscl_acc)
    baseline_stats = compute_statistics(baseline_acc)

    diff = abs(tscl_stats["mean"] - baseline_stats["mean"])

    return {
        "hypothesis": "H6: TSCL maintains task performance (within 1%)",
        "tscl_pass@1": tscl_stats,
        "baseline_pass@1": baseline_stats,
        "absolute_difference": diff,
        "supported": diff <= 0.01,
    }


def main(args):
    results_by_method = load_results(args.output_dir)

    print("=" * 60)
    print("TSCL EXPERIMENT RESULTS ANALYSIS")
    print("=" * 60)

    print(f"\nFound {len(results_by_method)} methods:")
    for method, results in results_by_method.items():
        print(f"  - {method}: {len(results)} runs")

    print("\n" + "=" * 60)
    print("HYPOTHESIS VERIFICATION")
    print("=" * 60)

    hypotheses = [
        analyze_hypothesis_h1(results_by_method),
        analyze_hypothesis_h2(results_by_method),
        analyze_hypothesis_h3(results_by_method),
        analyze_hypothesis_h4(results_by_method),
        analyze_hypothesis_h5(results_by_method),
        analyze_hypothesis_h6(results_by_method),
    ]

    for i, h in enumerate(hypotheses, 1):
        print(f"\n[H{i}] {h['hypothesis']}")
        print(f"    Supported: {'✓ YES' if h.get('supported') else '✗ NO'}")
        for k, v in h.items():
            if k not in ["hypothesis", "supported"]:
                print(f"    {k}: {v}")

    all_results = {
        "methods": {
            method: {
                "n_runs": len(results),
                "final_metrics_summary": {
                    metric: compute_statistics(extract_final_metrics(results, metric))
                    for metric in ["loss", "pass@1", "pass@5", "pass@10",
                                   "strategy_surprise_mean", "success_surprise_mean",
                                   "strategy_cluster_entropy"]
                }
            }
            for method, results in results_by_method.items()
        },
        "hypotheses": hypotheses,
    }

    output_file = Path(args.output_dir) / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    main(args)
