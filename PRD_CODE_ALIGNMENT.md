# PRD: Code Alignment with Paper "SuS: Strategy-aware Surprise for Intrinsic Exploration"

## Executive Summary

This document defines the requirements and acceptance criteria for aligning the codebase with the paper arXiv:2601.10349. The current implementation has **fundamental algorithmic discrepancies** that must be resolved.

---

## Critical Discrepancies Identified

### 1. Strategy Stability (SS) - FUNDAMENTAL MISMATCH

| Aspect | Paper (Equation 2) | Current Code |
|--------|-------------------|--------------|
| **Definition** | `SS = sim(z_pre, z_post)` | `1 - cosine_sim(s_predicted, s_actual)` |
| **Meaning** | Temporal consistency between states BEFORE and AFTER action | Prediction error (inverted!) |
| **Inputs** | `z_pre = E(s_t)`, `z_post = E(s_{t+1})` | `s_predicted` from SPH, `s_actual` from PSE |
| **Interpretation** | High SS = maintained strategic coherence | Current: High value = LOW similarity |

**Impact**: The code computes the INVERSE of what the paper describes, and uses different inputs entirely.

### 2. Strategy Surprise (SuS) - FUNDAMENTAL MISMATCH

| Aspect | Paper (Equation 3) | Current Code |
|--------|-------------------|--------------|
| **Definition** | `SuS = \|s_hat_{t+1} - s_{t+1}\| * (1 - SS)` | `\|correctness - p_predicted_success\|` |
| **Components** | World model prediction error weighted by strategy instability | Simple success prediction error |
| **Requires** | World model M, strategy embeddings | Binary correctness, success prediction |

**Impact**: Completely different computation. Code has NO world model M.

### 3. Missing Architectural Components

| Component | Paper Description | Code Status |
|-----------|------------------|-------------|
| World Model M | "Predicts next state s_hat_{t+1} from (s_t, a_t)" | **MISSING** |
| Contrastive Learning | "InfoNCE loss with in-batch negatives" | **MISSING** (uses MSE+BCE) |
| Momentum Encoder | "MoCo-style stable target embeddings" | **MISSING** |
| Temperature Scaling | "Temperature parameter τ for contrastive loss" | **MISSING** |
| Meta-learned λ | "Meta-learned coefficients" | **MISSING** (fixed values) |

### 4. Architecture Mismatch

| Component | Paper | Code |
|-----------|-------|------|
| MLP Layers | 3 layers | 2 layers |
| Activation | ReLU | GELU |
| Normalization | Layer Normalization | None |

### 5. Hyperparameter Mismatch

| Parameter | Paper (Optimal) | Code (Default) |
|-----------|-----------------|----------------|
| λ_SS | 1.0 | 0.5 |
| λ_SuS | 0.0 | 0.5 |

**Note**: Paper claims λ_SuS = 0.0 is optimal, which means the "SuS" component adds nothing. This is a significant finding.

---

## Acceptance Criteria

### AC-1: Strategy Stability (SS) Implementation

**MUST**:
- [ ] Compute SS as cosine similarity (NOT 1 - similarity)
- [ ] SS inputs must be embeddings from BEFORE action (query state) and AFTER action (response state)
- [ ] Use proper temporal structure: `SS(s_t, a_t, s_{t+1}) = sim(E(s_t), E(s_{t+1}))`
- [ ] SS range: [0, 1] where 1 = perfect strategic coherence

**Verification**:
```python
# Test case: identical states should have SS ≈ 1.0
assert compute_SS(same_embedding, same_embedding) > 0.99
# Test case: orthogonal states should have SS ≈ 0.0
assert compute_SS(embedding_a, orthogonal_embedding) < 0.1
```

### AC-2: Strategy Surprise (SuS) Implementation

**MUST**:
- [ ] Implement World Model M that predicts next state: `s_hat_{t+1} = M(s_t, a_t)`
- [ ] Compute SuS as: `SuS = ||s_hat_{t+1} - s_{t+1}|| * (1 - SS)`
- [ ] World model should be a separate trainable module
- [ ] SuS should be weighted by strategy instability factor `(1 - SS)`

**Verification**:
```python
# Test: When SS=1 (stable), SuS should be 0 regardless of prediction error
assert compute_SuS(prediction_error=0.5, SS=1.0) == 0.0
# Test: When SS=0 (unstable), SuS equals raw prediction error
assert compute_SuS(prediction_error=0.5, SS=0.0) == 0.5
```

### AC-3: World Model Architecture

**MUST**:
- [ ] Implement world model M with architecture: `M(s_t, a_t) -> s_hat_{t+1}`
- [ ] Input: state embedding + action embedding
- [ ] Output: predicted next state embedding
- [ ] Three-layer MLP with ReLU activations
- [ ] Include layer normalization as per paper
- [ ] Action embedding layer as described in paper

**Architecture**:
```python
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.action_embedding = nn.Linear(action_dim, state_dim)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, s_t, a_t):
        a_emb = self.action_embedding(a_t)
        combined = torch.cat([s_t, a_emb], dim=-1)
        return self.mlp(combined)
```

### AC-4: Contrastive Learning Framework

**MUST**:
- [ ] Replace MSE + BCE loss with InfoNCE contrastive loss
- [ ] Implement in-batch negatives
- [ ] Add temperature parameter τ (configurable)
- [ ] Implement momentum encoder for stable target embeddings

**InfoNCE Loss**:
```python
def info_nce_loss(query, positive, negatives, temperature=0.07):
    # query: anchor embedding
    # positive: positive sample embedding
    # negatives: batch of negative embeddings
    pos_sim = F.cosine_similarity(query, positive) / temperature
    neg_sim = F.cosine_similarity(query.unsqueeze(1), negatives) / temperature
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)
```

### AC-5: Momentum Encoder

**MUST**:
- [ ] Implement MoCo-style momentum encoder
- [ ] Exponential moving average update: `θ_k = m * θ_k + (1-m) * θ_q`
- [ ] Default momentum coefficient m = 0.999
- [ ] Use momentum encoder for target embeddings (stable representations)

### AC-6: Network Architecture Updates

**MUST**:
- [ ] Update Strategy Prediction Head to 3 layers
- [ ] Replace GELU with ReLU activations
- [ ] Add LayerNorm after each linear layer (before activation)

**Updated Architecture**:
```python
self.strategy_predictor = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.LayerNorm(hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, strategy_dim),
)
```

### AC-7: Intrinsic Reward Formula

**MUST**:
- [ ] Correct formula: `r_int = λ_SS * SS + λ_SuS * SuS`
- [ ] Note: SS is similarity (high = good), not dissimilarity
- [ ] Update default hyperparameters to match paper optimal values
- [ ] λ_SS = 1.0, λ_SuS = 0.0 (can be overridden)

### AC-8: Configuration Updates

**MUST**:
- [ ] Add new hyperparameters:
  - `temperature` for contrastive loss (default: 0.07)
  - `momentum` for momentum encoder (default: 0.999)
  - `world_model_hidden_dim` (default: 256)
- [ ] Update default λ values to match paper
- [ ] Add toggle for contrastive vs MSE learning mode

### AC-9: Naming Consistency

**MUST**:
- [ ] Rename `strategy_surprise` variable to `strategy_stability` (SS)
- [ ] Rename `success_surprise` variable to `strategy_surprise` (SuS)
- [ ] Rename `lambda_strategy` to `lambda_ss`
- [ ] Rename `lambda_success` to `lambda_sus`
- [ ] Update all references in code and configs

### AC-10: Backwards Compatibility (OPTIONAL)

**MAY**:
- [ ] Keep legacy mode for reproducibility of existing experiments
- [ ] Add `legacy_mode: true/false` config option
- [ ] Document differences between modes

---

## Implementation Priority

### Phase 1: Core Algorithm Fixes (CRITICAL)
1. Fix SS computation (similarity, not dissimilarity)
2. Implement World Model M
3. Fix SuS computation with world model
4. Update naming for consistency

### Phase 2: Architecture Updates (HIGH)
1. Update MLP to 3 layers with ReLU + LayerNorm
2. Implement momentum encoder
3. Add action embedding for world model

### Phase 3: Training Updates (MEDIUM)
1. Implement InfoNCE contrastive loss
2. Add temperature scaling
3. Implement in-batch negatives

### Phase 4: Configuration & Cleanup (LOW)
1. Update config files with new parameters
2. Update default hyperparameters
3. Update documentation/README

---

## Test Plan

### Unit Tests

1. **SS Computation Test**:
   - Identical embeddings -> SS ≈ 1.0
   - Orthogonal embeddings -> SS ≈ 0.0
   - Random embeddings -> 0 < SS < 1

2. **SuS Computation Test**:
   - When SS = 1.0 -> SuS = 0 (regardless of prediction error)
   - When SS = 0.0 -> SuS = ||prediction_error||
   - Verify weighting by (1 - SS) factor

3. **World Model Test**:
   - Output shape matches state dimension
   - Gradients flow properly
   - Training reduces prediction error

4. **Contrastive Loss Test**:
   - Positive pairs have lower loss than negative pairs
   - Temperature scaling affects gradient magnitude

### Integration Tests

1. **Full Training Loop**:
   - Model trains without errors
   - Losses decrease over time
   - Metrics are computed correctly

2. **Ablation Verification**:
   - SS-only mode works (λ_SuS = 0)
   - SuS-only mode works (λ_SS = 0)
   - Combined mode works

---

## Risk Assessment

### High Risk
- **World Model Training Stability**: New component may be hard to train
- **Contrastive Learning Sensitivity**: Temperature, batch size critical

### Medium Risk
- **Computational Overhead**: Momentum encoder + world model add computation
- **Hyperparameter Sensitivity**: New parameters may require tuning

### Low Risk
- **Naming Changes**: Breaking change for existing config files

---

## Success Metrics

1. **Algorithmic Correctness**: All formulas match paper exactly
2. **Test Coverage**: All acceptance criteria have passing tests
3. **Reproducibility**: Results match paper Table 3 within reasonable variance
4. **Documentation**: README accurately describes implementation

---

## Open Questions

1. **Paper claims λ_SuS = 0.0 is optimal** - should we still implement SuS or focus on SS only?
2. **Action representation**: How to embed action (generated text) for world model input?
3. **Trajectory structure**: How to define s_t and s_{t+1} in LLM generation context?

---

## Appendix: Formula Reference

### Paper Equations

**Equation 1** - Augmented Reward:
```
r = r_ext + α * r_int
```

**Equation 2** - Strategy Stability:
```
SS(s_t, a_t, s_{t+1}) = sim(z_pre, z_post) = (z_pre · z_post) / (||z_pre|| ||z_post||)
```

**Equation 3** - Strategy Surprise:
```
SuS(s_t, a_t, s_{t+1}) = ||s_hat_{t+1} - s_{t+1}|| * (1 - SS(s_t, a_t, s_{t+1}))
```

**Equation 4** - Intrinsic Reward:
```
r_int = λ_SS * SS + λ_SuS * SuS
```

### Current Code (INCORRECT)
```python
# This is WRONG - computes inverse of SS
strategy_surprise = 1 - F.cosine_similarity(s_predicted, s_actual)

# This is WRONG - has nothing to do with strategy or world model
success_surprise = torch.abs(correctness - p_predicted_success)
```

### Required Code (CORRECT)
```python
# Correct SS - temporal stability
strategy_stability = F.cosine_similarity(z_pre, z_post)

# Correct SuS - world model prediction error weighted by instability
s_hat_next = world_model(z_pre, action_embedding)
prediction_error = torch.norm(s_hat_next - z_post, dim=-1)
strategy_surprise = prediction_error * (1 - strategy_stability)

# Correct intrinsic reward
r_int = lambda_ss * strategy_stability + lambda_sus * strategy_surprise
```
