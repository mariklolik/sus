# SuS: Strategy-aware Surprise for Intrinsic Exploration

Official implementation for the paper "SuS: Strategy-aware Surprise for Intrinsic Exploration".

## Overview

SuS introduces two complementary intrinsic reward components:
- **Strategy Stability (SS)**: Measures consistency in behavioral strategy across transitions
- **Strategy Surprise (SuS)**: Captures unexpected outcomes relative to the agent's strategy representation

```
r_int = λ₁·SS + λ₂·SuS
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from sus import SuSReward

sus = SuSReward(strategy_dim=128, lambda_ss=1.0, lambda_sus=0.5)

# During training loop
z_pre = sus.encode(state)
z_post = sus.encode(next_state)
intrinsic_reward = sus.compute_reward(z_pre, z_post, prediction_error)
```

## Experiments

Train on GSM8K:

```bash
python train.py --config configs/gsm8k.yaml
```

Run ablations:

```bash
python train.py --config configs/ablation_ss_only.yaml
python train.py --config configs/ablation_sus_only.yaml
```

## Results

| Method | Pass@1 | Pass@5 |
|--------|--------|--------|
| SuS (Ours) | **17.8%** | **58.0%** |
| Baseline | 12.8% | 39.8% |
| SS Only | 14.5% | 45.2% |
| SuS Only | 15.2% | 48.5% |

## Citation

```bibtex
@article{kashirskiy2026sus,
  title={SuS: Strategy-aware Surprise for Intrinsic Exploration},
  author={Kashirskiy, Mark and Makarov, Ilya},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
