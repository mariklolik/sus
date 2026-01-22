#!/usr/bin/env python3
"""
Minimal pipeline test for SuS implementation.
Runs on CPU with tiny model to verify the full training loop works.
"""

import torch
import torch.nn.functional as F
import yaml
import sys
from pathlib import Path

# Force CPU
device = "cpu"


def test_full_pipeline():
    """Test the full SuS training pipeline."""
    print("\n" + "="*60)
    print("Testing Full SuS Training Pipeline (CPU)")
    print("="*60 + "\n")

    # Import after setting device
    from model import (
        SuSTrainer, WorldModel, StrategyPredictionHead,
        PostHocStrategyEncoder, MomentumEncoder, SuSRewardModule,
        InfoNCELoss, GRPOLoss
    )

    # Load test config
    config_path = Path(__file__).parent / "config_test_cpu.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("1. Testing component initialization...")

    hidden_dim = 768  # GPT-2 hidden size
    strategy_dim = config["tscl"]["strategy_dim"]

    # Initialize components
    sph = StrategyPredictionHead(hidden_dim, strategy_dim)
    print("   ✓ StrategyPredictionHead initialized")

    world_model = WorldModel(strategy_dim, strategy_dim, hidden_dim=64)
    print("   ✓ WorldModel initialized")

    pse = PostHocStrategyEncoder(
        model_name=config["tscl"]["pse_model"],
        strategy_dim=strategy_dim,
    )
    print("   ✓ PostHocStrategyEncoder initialized")

    momentum_encoder = MomentumEncoder(sph.strategy_predictor, momentum=0.999)
    print("   ✓ MomentumEncoder initialized")

    reward_module = SuSRewardModule(
        lambda_ss=config["tscl"]["lambda_ss"],
        lambda_sus=config["tscl"]["lambda_sus"],
    )
    print("   ✓ SuSRewardModule initialized")

    info_nce = InfoNCELoss(temperature=config["tscl"]["temperature"])
    print("   ✓ InfoNCELoss initialized")

    grpo_loss = GRPOLoss()
    print("   ✓ GRPOLoss initialized")

    print("\n2. Testing forward passes...")

    batch_size = 2

    # Simulate hidden states from LLM
    query_hidden = torch.randn(batch_size, hidden_dim)
    response_hidden = torch.randn(batch_size, hidden_dim)

    # SPH forward
    s_pred, p_success = sph(query_hidden)
    assert s_pred.shape == (batch_size, strategy_dim)
    assert p_success.shape == (batch_size,)
    print("   ✓ SPH forward pass OK")

    # PSE forward
    texts = ["Test response 1", "Test response 2"]
    z_post = pse(texts)
    assert z_post.shape == (batch_size, strategy_dim)
    print("   ✓ PSE forward pass OK")

    # World model forward
    s_hat_next = world_model(s_pred, z_post)
    assert s_hat_next.shape == (batch_size, strategy_dim)
    print("   ✓ WorldModel forward pass OK")

    # Reward module forward
    correctness = torch.tensor([1, 0], dtype=torch.float)
    intrinsic_reward, metrics = reward_module(
        z_pre=s_pred,
        z_post=z_post,
        s_hat_next=s_hat_next,
        correctness=correctness,
    )
    assert intrinsic_reward.shape == (batch_size,)
    print("   ✓ SuSRewardModule forward pass OK")
    print(f"     - Strategy Stability (SS): {metrics['strategy_stability_mean']:.4f}")
    print(f"     - Strategy Surprise (SuS): {metrics['strategy_surprise_mean']:.4f}")
    print(f"     - Intrinsic Reward: {metrics['intrinsic_reward_mean']:.4f}")

    # InfoNCE loss
    contrastive_loss = info_nce(s_pred, z_post)
    print(f"   ✓ InfoNCE loss computed: {contrastive_loss.item():.4f}")

    # GRPO loss
    logprobs = torch.randn(batch_size)
    ref_logprobs = torch.randn(batch_size)
    rewards = torch.tensor([1.0, -1.0])

    policy_loss, grpo_metrics = grpo_loss(logprobs, ref_logprobs, rewards)
    print(f"   ✓ GRPO loss computed: {policy_loss.item():.4f}")

    print("\n3. Testing gradient flow...")

    # Zero gradients
    sph.zero_grad()
    world_model.zero_grad()

    # Forward pass
    s_pred, _ = sph(query_hidden)
    s_hat = world_model(s_pred, z_post.detach())

    # Compute loss
    loss = F.mse_loss(s_hat, z_post.detach())

    # Backward pass
    loss.backward()

    # Check gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in sph.parameters())
    assert has_grad, "SPH should have gradients"
    print("   ✓ SPH gradients flow correctly")

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in world_model.parameters())
    assert has_grad, "WorldModel should have gradients"
    print("   ✓ WorldModel gradients flow correctly")

    print("\n4. Testing momentum encoder update...")

    # Store original weights
    orig_weight = momentum_encoder.encoder[0].weight.clone()

    # Update base encoder
    with torch.no_grad():
        for p in sph.strategy_predictor.parameters():
            p.add_(0.1)

    # Update momentum encoder
    momentum_encoder.update(sph.strategy_predictor)

    # Check weights changed
    new_weight = momentum_encoder.encoder[0].weight
    assert not torch.allclose(orig_weight, new_weight), \
        "Momentum encoder should update"
    print("   ✓ Momentum encoder EMA update works")

    print("\n5. Verifying paper equations implementation...")

    # Equation 2: SS = cosine_similarity(z_pre, z_post)
    z_same = F.normalize(torch.randn(1, strategy_dim), dim=-1)
    reward, metrics = reward_module(z_same, z_same, z_same, torch.ones(1))
    ss = metrics["strategy_stability_mean"]
    assert ss > 0.99, f"SS for identical vectors should be ~1.0, got {ss}"
    print(f"   ✓ Equation 2 (SS): cosine_similarity verified (SS={ss:.4f})")

    # Equation 3: SuS = ||s_hat - s|| * (1 - SS)
    # When SS=1, SuS should be 0
    sus = metrics["strategy_surprise_mean"]
    assert sus < 0.01, f"SuS should be ~0 when SS=1, got {sus}"
    print(f"   ✓ Equation 3 (SuS): prediction_error * (1-SS) verified (SuS={sus:.4f})")

    # Equation 4: r_int = lambda_ss * SS + lambda_sus * SuS
    # With lambda_ss=1, lambda_sus=0, r_int should equal SS for correct answers
    r_int = metrics["intrinsic_reward_mean"]
    assert abs(r_int - ss) < 0.01, f"r_int should equal SS, got {r_int} vs {ss}"
    print(f"   ✓ Equation 4 (r_int): lambda_ss*SS + lambda_sus*SuS verified")

    print("\n" + "="*60)
    print("All Pipeline Tests PASSED!")
    print("="*60 + "\n")

    return True


def test_ablation_configs():
    """Verify ablation configs have correct lambda values."""
    print("\n" + "="*60)
    print("Verifying Ablation Configurations")
    print("="*60 + "\n")

    configs_to_check = [
        ("config.yaml", "Full SuS", {"lambda_ss": 1.0, "lambda_sus": 0.0}),
        ("config_ablation_no_sus.yaml", "SS Only (No SuS)", {"lambda_ss": 1.0, "lambda_sus": 0.0}),
        ("config_ablation_no_ss.yaml", "SuS Only (No SS)", {"lambda_ss": 0.0, "lambda_sus": 1.0}),
        ("config_baseline.yaml", "Baseline (Disabled)", {"enabled": False}),
    ]

    src_dir = Path(__file__).parent

    for filename, description, expected in configs_to_check:
        filepath = src_dir / filename
        if not filepath.exists():
            print(f"   ⚠ {filename} not found")
            continue

        with open(filepath) as f:
            config = yaml.safe_load(f)

        tscl = config.get("tscl", {})

        if "enabled" in expected:
            actual_enabled = tscl.get("enabled", True)
            status = "✓" if actual_enabled == expected["enabled"] else "✗"
            print(f"   {status} {filename}: enabled={actual_enabled}")
        else:
            lambda_ss = tscl.get("lambda_ss", 0.5)
            lambda_sus = tscl.get("lambda_sus", 0.5)

            ss_ok = abs(lambda_ss - expected["lambda_ss"]) < 0.01
            sus_ok = abs(lambda_sus - expected["lambda_sus"]) < 0.01

            status = "✓" if (ss_ok and sus_ok) else "✗"
            print(f"   {status} {filename} ({description}):")
            print(f"      lambda_ss={lambda_ss} (expected {expected['lambda_ss']})")
            print(f"      lambda_sus={lambda_sus} (expected {expected['lambda_sus']})")

    print("\nNote: Paper claims optimal values are lambda_ss=1.0, lambda_sus=0.0")
    print("This means Strategy Surprise (SuS) component is NOT useful by paper's own results!")

    return True


if __name__ == "__main__":
    try:
        success = test_full_pipeline()
        test_ablation_configs()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
