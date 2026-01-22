#!/usr/bin/env python3
"""
Ablation Study: Verify optimal lambda_ss and lambda_sus values.

Tests the paper's claim that lambda_sus = 0.0 is optimal.
Runs multiple configurations and compares performance.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Force CPU if no GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from model import (
    SuSTrainer, WorldModel, StrategyPredictionHead,
    PostHocStrategyEncoder, SuSRewardModule, GRPOLoss
)


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""
    name: str
    lambda_ss: float
    lambda_sus: float
    description: str


# Ablation configurations to test
ABLATION_CONFIGS = [
    AblationConfig("baseline", 0.0, 0.0, "No intrinsic reward"),
    AblationConfig("ss_only", 1.0, 0.0, "SS only (paper optimal)"),
    AblationConfig("sus_only", 0.0, 1.0, "SuS only"),
    AblationConfig("balanced", 0.5, 0.5, "Balanced SS + SuS"),
    AblationConfig("ss_dominant", 0.8, 0.2, "SS dominant"),
    AblationConfig("sus_dominant", 0.2, 0.8, "SuS dominant"),
    AblationConfig("full_both", 1.0, 1.0, "Full contribution from both"),
]


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_data(
    num_problems: int = 50,
    num_trajectories: int = 4,
    hidden_dim: int = 256,
    strategy_dim: int = 128,
) -> Dict:
    """
    Create synthetic data that simulates the training scenario.

    We create:
    - Query embeddings (z_pre) - representing problem states
    - Response embeddings (z_post) - representing solution strategies
    - Correctness labels
    - Rewards based on correctness
    """
    data = []

    for problem_idx in range(num_problems):
        # Create a "problem embedding" - consistent for all trajectories of same problem
        problem_embedding = F.normalize(torch.randn(1, hidden_dim), dim=-1)

        for traj_idx in range(num_trajectories):
            # Query hidden state - slight variation from problem embedding
            query_hidden = problem_embedding + 0.1 * torch.randn(1, hidden_dim)
            query_hidden = F.normalize(query_hidden, dim=-1)

            # Simulate different solution strategies with varying correctness
            # More diverse strategies should lead to more exploration
            if random.random() < 0.3:  # 30% correct
                # Correct solutions tend to have consistent strategy
                response_hidden = query_hidden + 0.2 * torch.randn(1, hidden_dim)
                correctness = 1
                reward = 1.0
            else:
                # Incorrect solutions have more random strategies
                response_hidden = F.normalize(torch.randn(1, hidden_dim), dim=-1)
                correctness = 0
                reward = -1.0

            data.append({
                "query_hidden": query_hidden.squeeze(0),
                "response_hidden": response_hidden.squeeze(0),
                "correctness": correctness,
                "reward": reward,
                "problem_idx": problem_idx,
            })

    return data


def evaluate_config(
    config: AblationConfig,
    data: List[Dict],
    hidden_dim: int = 256,
    strategy_dim: int = 128,
    num_epochs: int = 10,
    batch_size: int = 16,
    seed: int = 42,
) -> Dict:
    """
    Evaluate a single ablation configuration.

    Returns metrics including:
    - Average intrinsic reward
    - Strategy diversity (cluster entropy)
    - Correlation between reward and correctness
    - Training loss progression
    """
    set_seed(seed)

    # Initialize components
    sph = StrategyPredictionHead(hidden_dim, strategy_dim).to(device)
    world_model = WorldModel(strategy_dim, strategy_dim, hidden_dim=128).to(device)
    pse_projector = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, strategy_dim),
        torch.nn.LayerNorm(strategy_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(strategy_dim, strategy_dim),
    ).to(device)

    reward_module = SuSRewardModule(
        lambda_ss=config.lambda_ss,
        lambda_sus=config.lambda_sus,
        only_reward_correct=False,  # Evaluate on all samples
    )

    # Optimizer
    params = list(sph.parameters()) + list(world_model.parameters()) + list(pse_projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    # Training metrics
    epoch_metrics = []
    all_intrinsic_rewards = []
    all_ss_values = []
    all_sus_values = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_rewards = []
        epoch_ss = []
        epoch_sus = []
        epoch_correct_rewards = []
        epoch_incorrect_rewards = []

        # Shuffle data
        indices = list(range(len(data)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [data[idx] for idx in batch_indices]

            # Prepare batch tensors
            query_hidden = torch.stack([d["query_hidden"] for d in batch]).to(device)
            response_hidden = torch.stack([d["response_hidden"] for d in batch]).to(device)
            correctness = torch.tensor([d["correctness"] for d in batch], dtype=torch.float).to(device)
            rewards = torch.tensor([d["reward"] for d in batch], dtype=torch.float).to(device)

            # Forward pass through SPH
            s_pred, p_success = sph(query_hidden)

            # Simulate PSE output (normally from text)
            z_post = F.normalize(pse_projector(response_hidden), dim=-1)

            # World model prediction
            s_hat_next = world_model(s_pred, z_post)

            # Compute intrinsic reward
            intrinsic_reward, metrics = reward_module(
                z_pre=s_pred,
                z_post=z_post,
                s_hat_next=s_hat_next,
                correctness=correctness,
            )

            # Training loss: encourage SPH to predict good strategies
            # and world model to predict next states
            strategy_loss = F.mse_loss(s_pred, z_post.detach())
            world_model_loss = F.mse_loss(s_hat_next, z_post.detach())
            success_loss = F.binary_cross_entropy(p_success, correctness)

            total_loss = strategy_loss + world_model_loss + success_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # Record metrics
            epoch_rewards.extend(intrinsic_reward.detach().cpu().numpy().tolist())
            epoch_ss.append(metrics["strategy_stability_mean"])
            epoch_sus.append(metrics["strategy_surprise_mean"])

            # Separate rewards by correctness
            for j, c in enumerate(correctness.cpu().numpy()):
                if c == 1:
                    epoch_correct_rewards.append(intrinsic_reward[j].item())
                else:
                    epoch_incorrect_rewards.append(intrinsic_reward[j].item())

        # Compute epoch statistics
        avg_reward = np.mean(epoch_rewards)
        avg_ss = np.mean(epoch_ss)
        avg_sus = np.mean(epoch_sus)
        avg_correct_reward = np.mean(epoch_correct_rewards) if epoch_correct_rewards else 0
        avg_incorrect_reward = np.mean(epoch_incorrect_rewards) if epoch_incorrect_rewards else 0

        # Reward discrimination: how well does intrinsic reward distinguish correct vs incorrect
        reward_discrimination = avg_correct_reward - avg_incorrect_reward

        epoch_metrics.append({
            "epoch": epoch + 1,
            "loss": epoch_loss / (len(indices) // batch_size),
            "avg_intrinsic_reward": avg_reward,
            "avg_ss": avg_ss,
            "avg_sus": avg_sus,
            "avg_correct_reward": avg_correct_reward,
            "avg_incorrect_reward": avg_incorrect_reward,
            "reward_discrimination": reward_discrimination,
        })

        all_intrinsic_rewards.extend(epoch_rewards)
        all_ss_values.append(avg_ss)
        all_sus_values.append(avg_sus)

    # Compute final metrics
    final_metrics = {
        "config_name": config.name,
        "lambda_ss": config.lambda_ss,
        "lambda_sus": config.lambda_sus,
        "description": config.description,
        "final_avg_reward": np.mean(all_intrinsic_rewards[-len(data):]),
        "final_avg_ss": np.mean(all_ss_values[-3:]),
        "final_avg_sus": np.mean(all_sus_values[-3:]),
        "final_loss": epoch_metrics[-1]["loss"],
        "final_reward_discrimination": epoch_metrics[-1]["reward_discrimination"],
        "reward_std": np.std(all_intrinsic_rewards[-len(data):]),
        "epoch_metrics": epoch_metrics,
    }

    return final_metrics


def compute_strategy_diversity(embeddings: torch.Tensor) -> float:
    """Compute strategy diversity using cluster entropy."""
    from sklearn.cluster import KMeans

    emb_np = embeddings.detach().cpu().numpy()
    n_clusters = min(4, len(emb_np) // 2)

    if n_clusters < 2:
        return 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    labels = kmeans.fit_predict(emb_np)

    counts = np.bincount(labels, minlength=n_clusters)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return entropy


def run_ablation_study(
    num_problems: int = 100,
    num_trajectories: int = 4,
    num_epochs: int = 15,
    seeds: List[int] = [42, 123, 456],
) -> Dict:
    """
    Run full ablation study across all configurations and seeds.
    """
    print("\n" + "="*70)
    print("SuS Ablation Study: Verifying Optimal Lambda Values")
    print("="*70)
    print(f"\nExperimental Setup:")
    print(f"  - Problems: {num_problems}")
    print(f"  - Trajectories per problem: {num_trajectories}")
    print(f"  - Training epochs: {num_epochs}")
    print(f"  - Seeds: {seeds}")
    print(f"  - Device: {device}")

    # Generate synthetic data (same across configs for fair comparison)
    print("\nGenerating synthetic training data...")
    all_data = {}
    for seed in seeds:
        set_seed(seed)
        all_data[seed] = create_synthetic_data(num_problems, num_trajectories)

    results = {}

    print("\nRunning ablation experiments...")
    print("-" * 70)

    for config in ABLATION_CONFIGS:
        print(f"\n[{config.name}] λ_SS={config.lambda_ss}, λ_SuS={config.lambda_sus}")
        print(f"  Description: {config.description}")

        config_results = []

        for seed in tqdm(seeds, desc=f"  Seeds", leave=False):
            metrics = evaluate_config(
                config=config,
                data=all_data[seed],
                num_epochs=num_epochs,
                seed=seed,
            )
            config_results.append(metrics)

        # Aggregate across seeds
        avg_reward = np.mean([r["final_avg_reward"] for r in config_results])
        std_reward = np.std([r["final_avg_reward"] for r in config_results])
        avg_discrimination = np.mean([r["final_reward_discrimination"] for r in config_results])
        avg_ss = np.mean([r["final_avg_ss"] for r in config_results])
        avg_sus = np.mean([r["final_avg_sus"] for r in config_results])

        results[config.name] = {
            "config": {
                "lambda_ss": config.lambda_ss,
                "lambda_sus": config.lambda_sus,
                "description": config.description,
            },
            "metrics": {
                "avg_intrinsic_reward": avg_reward,
                "std_intrinsic_reward": std_reward,
                "avg_reward_discrimination": avg_discrimination,
                "avg_strategy_stability": avg_ss,
                "avg_strategy_surprise": avg_sus,
            },
            "seed_results": config_results,
        }

        print(f"  Results: reward={avg_reward:.4f}±{std_reward:.4f}, "
              f"discrimination={avg_discrimination:.4f}, SS={avg_ss:.4f}, SuS={avg_sus:.4f}")

    return results


def analyze_results(results: Dict) -> str:
    """Analyze and format ablation study results."""
    output = []
    output.append("\n" + "="*70)
    output.append("ABLATION STUDY RESULTS")
    output.append("="*70)

    # Sort by reward discrimination (key metric for usefulness)
    sorted_configs = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"]["avg_reward_discrimination"],
        reverse=True
    )

    output.append("\n### Ranking by Reward Discrimination")
    output.append("(Higher = better at distinguishing correct vs incorrect solutions)\n")

    output.append(f"{'Rank':<5} {'Config':<15} {'λ_SS':<6} {'λ_SuS':<6} {'Discrimination':<15} {'Avg Reward':<12}")
    output.append("-" * 70)

    for rank, (name, data) in enumerate(sorted_configs, 1):
        metrics = data["metrics"]
        config = data["config"]
        output.append(
            f"{rank:<5} {name:<15} {config['lambda_ss']:<6.1f} {config['lambda_sus']:<6.1f} "
            f"{metrics['avg_reward_discrimination']:<15.4f} {metrics['avg_intrinsic_reward']:<12.4f}"
        )

    # Analysis
    output.append("\n" + "-"*70)
    output.append("ANALYSIS")
    output.append("-"*70)

    best_config = sorted_configs[0]
    best_name = best_config[0]
    best_metrics = best_config[1]["metrics"]
    best_params = best_config[1]["config"]

    output.append(f"\n✓ Best configuration: {best_name}")
    output.append(f"  λ_SS = {best_params['lambda_ss']}, λ_SuS = {best_params['lambda_sus']}")
    output.append(f"  Reward discrimination: {best_metrics['avg_reward_discrimination']:.4f}")

    # Check if paper's claim holds
    ss_only = results.get("ss_only", {}).get("metrics", {})
    sus_only = results.get("sus_only", {}).get("metrics", {})
    balanced = results.get("balanced", {}).get("metrics", {})

    output.append("\n### Comparison of Key Configurations:")

    if ss_only and sus_only:
        ss_disc = ss_only.get("avg_reward_discrimination", 0)
        sus_disc = sus_only.get("avg_reward_discrimination", 0)

        output.append(f"\n  SS Only (λ_SS=1, λ_SuS=0):  discrimination = {ss_disc:.4f}")
        output.append(f"  SuS Only (λ_SS=0, λ_SuS=1): discrimination = {sus_disc:.4f}")

        if balanced:
            bal_disc = balanced.get("avg_reward_discrimination", 0)
            output.append(f"  Balanced (λ_SS=0.5, λ_SuS=0.5): discrimination = {bal_disc:.4f}")

        output.append("\n### Verdict on Paper's Claim:")

        if ss_disc > sus_disc and best_params['lambda_sus'] == 0.0:
            output.append("  ✓ Paper's claim SUPPORTED: λ_SuS=0.0 appears optimal")
            output.append("  The Strategy Surprise component does not improve performance")
        elif sus_disc > ss_disc:
            output.append("  ✗ Paper's claim CONTRADICTED: SuS component IS beneficial")
            output.append("  The Strategy Surprise component improves performance!")
        else:
            bal_disc = balanced.get("avg_reward_discrimination", 0) if balanced else 0
            if bal_disc > max(ss_disc, sus_disc):
                output.append("  ✗ Paper's claim CONTRADICTED: Combined SS+SuS is best")
                output.append("  Both components contribute to optimal performance!")
            else:
                output.append("  ~ Results inconclusive - need more experiments")

    return "\n".join(output)


def main():
    """Main entry point for ablation study."""
    # Run with reasonable parameters for CPU
    results = run_ablation_study(
        num_problems=100,
        num_trajectories=4,
        num_epochs=15,
        seeds=[42, 123, 456],
    )

    # Analyze results
    analysis = analyze_results(results)
    print(analysis)

    # Save results
    output_dir = Path("outputs_ablation")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    results_file = output_dir / f"ablation_results_{timestamp}.json"
    with open(results_file, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    # Save analysis
    analysis_file = output_dir / f"ablation_analysis_{timestamp}.txt"
    with open(analysis_file, "w") as f:
        f.write(analysis)

    print(f"\n✓ Results saved to {results_file}")
    print(f"✓ Analysis saved to {analysis_file}")

    return results


if __name__ == "__main__":
    main()
