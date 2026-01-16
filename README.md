# SuS: Strategy-aware Surprise for Intrinsic Exploration

Official implementation for the paper "SuS: Strategy-aware Surprise for Intrinsic Exploration".

[[Paper]]([https://arxiv.org/abs/XXXX.XXXXX](https://arxiv.org/abs/2601.10349)) [[Code]](https://github.com/mariklolik/sus)

## Overview

SuS introduces two complementary intrinsic reward components for exploration:

- **Strategy Stability (SS)**: Measures consistency in behavioral strategy across transitions
- **Strategy Surprise (SuS)**: Captures unexpected outcomes relative to the agent's strategy representation

The combined intrinsic reward:
```
r_int = λ_SS · SS + λ_SuS · SuS
```

## Installation

```bash
git clone https://github.com/mariklolik/sus.git
cd sus
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Full SuS training
python src/train.py --config src/config.yaml --method tscl

# Baseline (no intrinsic reward)
python src/train.py --config src/config_baseline.yaml --method baseline

# Ablations
python src/train.py --config src/config_ablation_no_ss.yaml --method tscl
python src/train.py --config src/config_ablation_no_sus.yaml --method tscl
```

### Run all experiments

```bash
python src/run_sequential.py
```

## Results

Results on GSM8K mathematical reasoning:

| Method | Pass@1 | Pass@5 | Entropy |
|--------|--------|--------|---------|
| **SuS (Ours)** | **14.2%** | **46.8%** | **1.31** |
| SS Only | 12.5% | 38.9% | 0.89 |
| SuS Only | 13.1% | 41.2% | 0.95 |
| Baseline | 12.1% | 37.1% | 0.65 |

## Project Structure

```
sus/
├── src/
│   ├── model.py          # SuS model implementation
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   ├── config.yaml       # Main config
│   └── config_*.yaml     # Ablation configs
├── configs/              # Additional configs
└── requirements.txt
```

## Citation

```bibtex
@article{kashirskiy2026sus,
  title={SuS: Strategy-aware Surprise for Intrinsic Exploration},
  author={Kashirskiy, Mark and Makarov, Ilya},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
