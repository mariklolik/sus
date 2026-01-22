#!/usr/bin/env python3
"""
Unit tests for SuS model components.
Verifies that the implementation matches paper equations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

# Test without GPU
device = "cpu"


def test_world_model():
    """Test WorldModel architecture and forward pass."""
    from model import WorldModel

    state_dim = 128
    action_dim = 128
    hidden_dim = 256
    batch_size = 4

    model = WorldModel(state_dim, action_dim, hidden_dim)

    # Check architecture: should be 3-layer MLP
    assert len([m for m in model.mlp if isinstance(m, torch.nn.Linear)]) == 3, \
        "WorldModel should have 3 linear layers"

    # Check LayerNorm presence
    assert len([m for m in model.mlp if isinstance(m, torch.nn.LayerNorm)]) == 2, \
        "WorldModel should have 2 LayerNorm layers"

    # Check ReLU (not GELU)
    assert len([m for m in model.mlp if isinstance(m, torch.nn.ReLU)]) == 2, \
        "WorldModel should use ReLU activations"

    # Forward pass
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)

    output = model(state, action)

    assert output.shape == (batch_size, state_dim), \
        f"WorldModel output shape mismatch: {output.shape}"

    # Output should be normalized
    norms = torch.norm(output, dim=-1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), \
        "WorldModel output should be L2 normalized"

    print("✓ WorldModel tests passed")


def test_strategy_prediction_head():
    """Test StrategyPredictionHead architecture."""
    from model import StrategyPredictionHead

    hidden_dim = 512
    strategy_dim = 128
    batch_size = 4

    sph = StrategyPredictionHead(hidden_dim, strategy_dim)

    # Check strategy_predictor: should be 3-layer MLP with ReLU and LayerNorm
    linear_layers = [m for m in sph.strategy_predictor if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3, \
        f"SPH strategy_predictor should have 3 linear layers, got {len(linear_layers)}"

    relu_layers = [m for m in sph.strategy_predictor if isinstance(m, torch.nn.ReLU)]
    assert len(relu_layers) == 2, \
        f"SPH should use ReLU activations, got {len(relu_layers)}"

    layernorm_layers = [m for m in sph.strategy_predictor if isinstance(m, torch.nn.LayerNorm)]
    assert len(layernorm_layers) == 2, \
        f"SPH should have LayerNorm, got {len(layernorm_layers)}"

    # Forward pass
    hidden = torch.randn(batch_size, hidden_dim)
    s_pred, p_success = sph(hidden)

    assert s_pred.shape == (batch_size, strategy_dim), \
        f"Strategy embedding shape mismatch: {s_pred.shape}"
    assert p_success.shape == (batch_size,), \
        f"Success probability shape mismatch: {p_success.shape}"

    # Strategy should be normalized
    norms = torch.norm(s_pred, dim=-1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), \
        "Strategy embedding should be L2 normalized"

    # Success probability should be in [0, 1]
    assert (p_success >= 0).all() and (p_success <= 1).all(), \
        "Success probability should be in [0, 1]"

    print("✓ StrategyPredictionHead tests passed")


def test_strategy_stability_formula():
    """
    Test Strategy Stability (SS) computation.

    Paper Equation 2:
    SS(s_t, a_t, s_{t+1}) = sim(z_pre, z_post) = cosine_similarity

    CRITICAL: SS should be cosine similarity (NOT 1 - similarity)
    """
    from model import SuSRewardModule

    batch_size = 4
    dim = 128

    reward_module = SuSRewardModule(lambda_ss=1.0, lambda_sus=0.0)

    # Test 1: Identical embeddings -> SS should be ~1.0
    z_identical = F.normalize(torch.randn(batch_size, dim), dim=-1)
    s_hat = z_identical.clone()  # Perfect prediction
    correctness = torch.ones(batch_size)

    reward, metrics = reward_module(
        z_pre=z_identical,
        z_post=z_identical,
        s_hat_next=s_hat,
        correctness=correctness,
    )

    ss_value = metrics["strategy_stability_mean"]
    assert ss_value > 0.99, \
        f"SS for identical embeddings should be ~1.0, got {ss_value}"

    # Test 2: Orthogonal embeddings -> SS should be ~0.0
    z_pre = torch.zeros(batch_size, dim)
    z_pre[:, 0] = 1.0  # Unit vector along first axis

    z_post = torch.zeros(batch_size, dim)
    z_post[:, 1] = 1.0  # Unit vector along second axis (orthogonal)

    reward, metrics = reward_module(
        z_pre=z_pre,
        z_post=z_post,
        s_hat_next=s_hat,
        correctness=correctness,
    )

    ss_value = metrics["strategy_stability_mean"]
    assert ss_value < 0.01, \
        f"SS for orthogonal embeddings should be ~0.0, got {ss_value}"

    # Test 3: CRITICAL - SS should NOT be inverted (old bug)
    # If the old code computed 1 - cosine_similarity, identical embeddings would give 0
    # This test ensures the bug is fixed
    z_same = F.normalize(torch.randn(1, dim), dim=-1)
    reward, metrics = reward_module(
        z_pre=z_same,
        z_post=z_same,
        s_hat_next=z_same,
        correctness=torch.ones(1),
    )

    ss_value = metrics["strategy_stability_mean"]
    assert ss_value > 0.5, \
        f"CRITICAL BUG: SS appears to be inverted! Got {ss_value} for identical embeddings"

    print("✓ Strategy Stability (SS) formula tests passed")


def test_strategy_surprise_formula():
    """
    Test Strategy Surprise (SuS) computation.

    Paper Equation 3:
    SuS = ||s_hat_{t+1} - s_{t+1}|| * (1 - SS)

    Key properties:
    - When SS = 1.0 (stable), SuS = 0 regardless of prediction error
    - When SS = 0.0 (unstable), SuS = ||prediction_error||
    """
    from model import SuSRewardModule

    batch_size = 1
    dim = 128

    reward_module = SuSRewardModule(lambda_ss=0.0, lambda_sus=1.0)

    # Test 1: When SS = 1 (identical embeddings), SuS should be 0
    z_identical = F.normalize(torch.randn(batch_size, dim), dim=-1)
    z_wrong_prediction = F.normalize(torch.randn(batch_size, dim), dim=-1)  # Wrong prediction

    reward, metrics = reward_module(
        z_pre=z_identical,
        z_post=z_identical,
        s_hat_next=z_wrong_prediction,  # Even with wrong prediction
        correctness=torch.ones(batch_size),
    )

    sus_value = metrics["strategy_surprise_mean"]
    # SuS = prediction_error * (1 - SS) = prediction_error * 0 = 0
    assert sus_value < 0.01, \
        f"SuS should be ~0 when SS=1, got {sus_value}"

    # Test 2: When SS = 0 (orthogonal embeddings), SuS = prediction error
    z_pre = torch.zeros(batch_size, dim)
    z_pre[:, 0] = 1.0

    z_post = torch.zeros(batch_size, dim)
    z_post[:, 1] = 1.0  # Orthogonal to z_pre

    # Make s_hat_next different from z_post to create prediction error
    s_hat_next = torch.zeros(batch_size, dim)
    s_hat_next[:, 2] = 1.0  # Different from z_post

    expected_error = torch.norm(s_hat_next - z_post, dim=-1).item()

    reward, metrics = reward_module(
        z_pre=z_pre,
        z_post=z_post,
        s_hat_next=s_hat_next,
        correctness=torch.ones(batch_size),
    )

    sus_value = metrics["strategy_surprise_mean"]
    # SuS = prediction_error * (1 - 0) = prediction_error
    assert abs(sus_value - expected_error) < 0.01, \
        f"SuS should equal prediction error ({expected_error}) when SS=0, got {sus_value}"

    print("✓ Strategy Surprise (SuS) formula tests passed")


def test_intrinsic_reward_formula():
    """
    Test intrinsic reward formula.

    Paper Equation 4:
    r_int = lambda_ss * SS + lambda_sus * SuS
    """
    from model import SuSRewardModule

    batch_size = 4
    dim = 128

    # Test with lambda_ss=1.0, lambda_sus=0.0 (paper optimal)
    reward_module = SuSRewardModule(lambda_ss=1.0, lambda_sus=0.0)

    z = F.normalize(torch.randn(batch_size, dim), dim=-1)
    s_hat = z.clone()
    correctness = torch.ones(batch_size)

    reward, metrics = reward_module(z, z, s_hat, correctness)

    # With identical embeddings: SS = 1, SuS = 0
    # r_int = 1.0 * 1 + 0.0 * 0 = 1.0
    expected_reward = 1.0
    actual_reward = metrics["intrinsic_reward_mean"]

    assert abs(actual_reward - expected_reward) < 0.01, \
        f"Intrinsic reward mismatch: expected {expected_reward}, got {actual_reward}"

    print("✓ Intrinsic reward formula tests passed")


def test_infonce_loss():
    """Test InfoNCE contrastive loss."""
    from model import InfoNCELoss

    batch_size = 8
    dim = 128

    info_nce = InfoNCELoss(temperature=0.07)

    # Positive pairs should have lower loss than negative pairs
    query = F.normalize(torch.randn(batch_size, dim), dim=-1)
    positive = query.clone()  # Perfect match

    loss_positive = info_nce(query, positive)

    # Random negatives
    negative = F.normalize(torch.randn(batch_size, dim), dim=-1)
    loss_negative = info_nce(query, negative)

    assert loss_positive < loss_negative, \
        f"Positive pairs should have lower loss: {loss_positive.item()} vs {loss_negative.item()}"

    print("✓ InfoNCE loss tests passed")


def test_momentum_encoder():
    """Test MomentumEncoder EMA update."""
    from model import MomentumEncoder
    import torch.nn as nn

    dim = 128
    momentum = 0.999

    base_encoder = nn.Linear(dim, dim)
    momentum_encoder = MomentumEncoder(base_encoder, momentum=momentum)

    # Store original weights
    original_weight = momentum_encoder.encoder.weight.clone()

    # Modify base encoder
    with torch.no_grad():
        base_encoder.weight.fill_(1.0)

    # Update momentum encoder
    momentum_encoder.update(base_encoder)

    # Check EMA update: theta_k = m * theta_k + (1-m) * theta_q
    expected_weight = momentum * original_weight + (1 - momentum) * torch.ones_like(original_weight)

    assert torch.allclose(momentum_encoder.encoder.weight, expected_weight, atol=1e-5), \
        "MomentumEncoder EMA update formula incorrect"

    print("✓ MomentumEncoder tests passed")


def test_naming_consistency():
    """Test that variable names match paper terminology."""
    from model import SuSOutput, SuSRewardModule

    # Check SuSOutput has correct field names
    output = SuSOutput(
        policy_loss=torch.tensor(0.0),
        intrinsic_loss=torch.tensor(0.0),
        total_loss=torch.tensor(0.0),
        strategy_stability=0.5,  # SS
        strategy_surprise=0.3,   # SuS
    )

    assert hasattr(output, 'strategy_stability'), "Missing strategy_stability field"
    assert hasattr(output, 'strategy_surprise'), "Missing strategy_surprise field"

    # Check SuSRewardModule uses correct parameter names
    module = SuSRewardModule(lambda_ss=1.0, lambda_sus=0.0)
    assert hasattr(module, 'lambda_ss'), "Missing lambda_ss parameter"
    assert hasattr(module, 'lambda_sus'), "Missing lambda_sus parameter"

    print("✓ Naming consistency tests passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("Running SuS Model Unit Tests")
    print("="*60 + "\n")

    tests = [
        test_world_model,
        test_strategy_prediction_head,
        test_strategy_stability_formula,
        test_strategy_surprise_formula,
        test_intrinsic_reward_formula,
        test_infonce_loss,
        test_momentum_encoder,
        test_naming_consistency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
