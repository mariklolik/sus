#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAX_SAMPLES=${MAX_SAMPLES:-200}
SEEDS=(42 123 456)

echo "=== TSCL Experiments ==="
echo "Max samples: $MAX_SAMPLES"
echo "Seeds: ${SEEDS[*]}"
echo ""

echo "=== Running Baseline (Vanilla RL) ==="
for seed in "${SEEDS[@]}"; do
    echo "Seed $seed..."
    python train.py --config config_baseline.yaml --seed $seed --method baseline --max_samples $MAX_SAMPLES --no_wandb
done

echo "=== Running TSCL (Full Method) ==="
for seed in "${SEEDS[@]}"; do
    echo "Seed $seed..."
    python train.py --config config.yaml --seed $seed --method tscl --max_samples $MAX_SAMPLES --no_wandb
done

echo "=== Running Perplexity Reward (CDE-style) ==="
for seed in "${SEEDS[@]}"; do
    echo "Seed $seed..."
    python train.py --config config.yaml --seed $seed --method perplexity --max_samples $MAX_SAMPLES --no_wandb
done

echo "=== Running Ablation: No Strategy Surprise ==="
for seed in "${SEEDS[@]}"; do
    echo "Seed $seed..."
    python train.py --config config_ablation_no_ss.yaml --seed $seed --method tscl --max_samples $MAX_SAMPLES --no_wandb
done

echo "=== Running Ablation: No Success Surprise ==="
for seed in "${SEEDS[@]}"; do
    echo "Seed $seed..."
    python train.py --config config_ablation_no_sus.yaml --seed $seed --method tscl --max_samples $MAX_SAMPLES --no_wandb
done

echo "=== All experiments complete ==="
