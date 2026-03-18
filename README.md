# SuS: Strategy-aware Surprise for Intrinsic Exploration

Official implementation for the paper **"SuS: Strategy-aware Surprise for Intrinsic Exploration"**.

[[Paper]](https://arxiv.org/abs/2601.10349) [[Code]](https://github.com/mariklolik/sus) [[Models]](https://huggingface.co/mariklolik228/sus-qwen2.5-1.5b-grpo-lora) [[Training Logs]](https://wandb.ai/mariklolikteam/sus-grpo)

## Overview

SuS introduces a strategy-aware intrinsic exploration reward for GRPO-based LLM fine-tuning. Standard GRPO relies solely on extrinsic correctness rewards and produces low-diversity solutions. SuS addresses this by adding a novelty bonus to correct responses in mixed-correctness batches, encouraging the model to discover diverse problem-solving strategies.

### Method

For each batch of $G$ generations per problem, SuS computes the reward as:

$$r_i = r_{\text{ext}}(i) + \mathbb{1}[\text{mixed}] \cdot \mathbb{1}[\text{correct}_i] \cdot \beta \cdot \text{novelty}(i) \cdot \text{difficulty}(q)$$

where:
- $r_{\text{ext}}(i)$ is the extrinsic reward (correctness + format)
- $\beta$ is the SS bonus coefficient (default: 0.1)
- $\text{novelty}(i) = 1 - \frac{1}{|\mathcal{C}|-1} \sum_{j \in \mathcal{C} \setminus i} \cos(\mathbf{e}_i, \mathbf{e}_j)$ measures inter-response strategy diversity using frozen SentenceTransformer embeddings $\mathbf{e}$
- $\text{difficulty}(q) = 1 - \text{EMA}_{\alpha}(\text{accuracy}(q))$ scales exploration toward harder problems
- $\mathcal{C}$ is the set of correct responses in the batch

**Key design principle:** zero-variance batches (all-correct or all-incorrect) are left untouched. This prevents the KL divergence blowup and length collapse that occurs when intrinsic rewards add per-sample noise to uniform batches.

## Installation

```bash
git clone https://github.com/mariklolik/sus.git
cd sus
pip install -r requirements.txt
```

## Training

```bash
# SuS (our method)
python src/train_sus.py --config src/config.yaml --ss_bonus 0.1 --seed 42

# Pure GRPO baseline
python src/train_grpo.py --config src/config.yaml --seed 42

# Baseline exploration methods
python src/train_baselines.py --config src/config.yaml --method rs_grpo --seed 42
python src/train_baselines.py --config src/config.yaml --method progrpo_arm --seed 42
python src/train_baselines.py --config src/config.yaml --method entropy_bonus --seed 42
python src/train_baselines.py --config src/config.yaml --method ceeh_difficulty --seed 42
python src/train_baselines.py --config src/config.yaml --method ucb_exploration --seed 42
```

## Evaluation

```bash
# Evaluate a specific checkpoint
python src/evaluate.py --method_name sus --checkpoint_dir outputs/grpo_sus_b01_e000_seed42 --n_samples 16

# Evaluate all default methods
python src/evaluate.py --output_dir outputs --n_samples 16

# Merge per-method results into a single file
python src/merge_eval_results.py
```

## Results

GSM8K test set (1319 problems), Qwen2.5-1.5B-Instruct + LoRA, 2000 GRPO steps, 16 samples/problem:

| Method | Pass@1 | Pass@5 | Pass@8 | 95% CI (Pass@1) |
|--------|--------|--------|--------|-----------------|
| **SuS (Ours, $\beta$=0.1)** | **74.42** | **90.36** | **92.74** | **[72.65, 76.22]** |
| GRPO Baseline | 73.98 | 89.53 | 91.88 | [72.10, 75.83] |
| RS-GRPO [(Li et al., 2025)](https://arxiv.org/abs/2509.24261) | 73.57 | 90.12 | 92.72 | [71.78, 75.36] |
| ProGRPO [(Zhao et al., 2025)](https://arxiv.org/abs/2602.05281) | 72.23 | 89.49 | 92.24 | [70.38, 74.04] |
| Entropy Bonus (EDGE-GRPO) | 61.37 | 81.80 | 85.71 | [59.29, 63.49] |

### Ablation: SS bonus coefficient $\beta$

| $\beta$ | err_bonus | Pass@1 | Pass@5 | Pass@8 |
|---------|-----------|--------|--------|--------|
| 0.0 | 0.0 | 73.46 | 90.07 | 92.57 |
| **0.1** | **0.0** | **74.42** | **90.36** | **92.74** |
| 0.2 | 0.0 | 73.11 | 89.91 | 92.43 |
| 0.2 | 0.05 | 72.06 | 89.47 | 92.10 |
| 0.3 | 0.05 | 72.28 | 89.54 | 92.37 |

## Pretrained Models

| Model | HuggingFace | Pass@1 |
|-------|-------------|--------|
| SuS (best, $\beta$=0.1) | [mariklolik228/sus-qwen2.5-1.5b-grpo-lora](https://huggingface.co/mariklolik228/sus-qwen2.5-1.5b-grpo-lora) | 74.42% |
| GRPO Baseline | [mariklolik228/grpo-baseline-qwen2.5-1.5b-lora](https://huggingface.co/mariklolik228/grpo-baseline-qwen2.5-1.5b-lora) | 73.98% |

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "mariklolik228/sus-qwen2.5-1.5b-grpo-lora")
tokenizer = AutoTokenizer.from_pretrained("mariklolik228/sus-qwen2.5-1.5b-grpo-lora")
```

## Project Structure

```
sus/
├── src/
│   ├── train_sus.py          # SuS training
│   ├── train_grpo.py         # Baseline GRPO training
│   ├── train_baselines.py    # Baseline exploration methods
│   ├── sus_reward.py         # SuS reward module
│   ├── baselines.py          # Baseline reward functions
│   ├── evaluate.py           # GSM8K evaluation with pass@k
│   ├── merge_eval_results.py # Merge eval result JSONs
│   ├── config.yaml           # Single-GPU training config
│   └── config_distributed.yaml # Multi-GPU config
├── paper/                    # LaTeX source
├── requirements.txt
└── LICENSE
```

## Citation

```bibtex
@article{kashirskiy2026sus,
  title={SuS: Strategy-aware Surprise for Intrinsic Exploration},
  author={Kashirskiy, Mark and Makarov, Ilya},
  journal={arXiv preprint arXiv:2601.10349},
  year={2026}
}
```

## License

MIT
