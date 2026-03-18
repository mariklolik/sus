#!/usr/bin/env python3
"""SuS (Strategy-aware Surprise) reward for TRL GRPOTrainer.

Combines correctness, format, and strategy novelty into a single reward function.
Key invariant: zero-variance batches (all-correct or all-incorrect) are preserved
to prevent spurious gradients and KL divergence blowup.
"""

import re
import math
import torch
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer


@dataclass
class SuSConfig:
    ss_bonus: float = 0.2
    err_bonus: float = 0.0
    difficulty_aware: bool = True
    difficulty_ema: float = 0.1
    pse_model: str = "all-MiniLM-L6-v2"
    random_pse: bool = False  # Use random unit vectors instead of real embeddings


def _extract_gsm8k_gt(ground_truth: str) -> Optional[str]:
    match = re.search(r"####\s*([\-\d,\.]+)", ground_truth)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def _extract_generated(text: str) -> Optional[str]:
    match = re.search(r"<answer>\s*([\-\d,\.]+)\s*</answer>", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"\\boxed\{([\-\d,\.]+)\}", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None


def _numbers_match(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-3
    except (ValueError, TypeError):
        return False


def _completion_to_text(completion) -> str:
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    elif isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


class PSEncoder:
    """Frozen SentenceTransformer for strategy embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu",
                 random_mode: bool = False):
        self.random_mode = random_mode
        self.device = device
        if not random_mode:
            self.encoder = SentenceTransformer(model_name, device=device)
        else:
            self.encoder = None
            self._dim = 384  # Match MiniLM-L6-v2 dimension

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.random_mode:
            # Random unit vectors — same shape as MiniLM, no semantic content
            embeddings = torch.randn(len(texts), self._dim)
            return F.normalize(embeddings.float(), dim=-1)
        embeddings = self.encoder.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        return F.normalize(embeddings.float(), dim=-1)


class DifficultyTracker:
    """EMA tracker of per-problem accuracy. Hard problems get more exploration."""

    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.problem_acc: Dict[str, float] = {}

    def get_difficulty(self, problem_id: str) -> float:
        ema = self.problem_acc.get(problem_id, 0.5)
        return 1.0 - ema

    def update(self, problem_id: str, accuracy: float):
        if problem_id in self.problem_acc:
            old = self.problem_acc[problem_id]
            self.problem_acc[problem_id] = (
                self.ema_alpha * accuracy + (1.0 - self.ema_alpha) * old
            )
        else:
            self.problem_acc[problem_id] = accuracy


class SuSReward:
    """Combined correctness + format + strategy novelty reward.

    Reward structure:
        all-incorrect batch:  0.0 + format (uniform -> zero gradient)
        all-correct batch:    1.0 + format (uniform -> zero gradient)
        mixed batch correct:  1.0 + format + ss_bonus * novelty * difficulty
        mixed batch incorrect: 0.0 + format [+ err_bonus * answer_novelty]
    """

    def __init__(self, config: SuSConfig, device: str = "cpu"):
        self.config = config
        self.pse = PSEncoder(model_name=config.pse_model, device=device,
                             random_mode=config.random_pse)
        self.difficulty = DifficultyTracker(ema_alpha=config.difficulty_ema)
        self.answer_counts: Dict[str, int] = defaultdict(int)
        self.total_answers: int = 0
        self._last_metrics: Dict[str, float] = {}

    def _compute_novelty_intra_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        n = embeddings.shape[0]
        if n <= 1:
            return torch.ones(n)
        sim_matrix = embeddings @ embeddings.T
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix * mask
        novelty = 1.0 - sim_matrix.sum(dim=1) / (n - 1)
        return novelty.clamp(0, 1)

    def _answer_novelty(self, answer: Optional[str]) -> float:
        if answer is None or self.total_answers == 0:
            return 1.0
        count = self.answer_counts.get(answer, 0)
        if count == 0:
            return 1.0
        return min(1.0, math.sqrt(math.log(self.total_answers + 1) / (count + 1)))

    def _process_group(
        self, prompt: str, completions: List[str], ground_truths: List[str],
    ) -> List[float]:
        texts = []
        correct_flags = []
        format_flags = []
        answers = []

        for completion, gt in zip(completions, ground_truths):
            text = _completion_to_text(completion)
            texts.append(text)
            gt_num = _extract_gsm8k_gt(gt)
            gen_num = _extract_generated(text)
            is_correct = (
                gt_num is not None
                and gen_num is not None
                and _numbers_match(gen_num, gt_num)
            )
            correct_flags.append(is_correct)
            format_flags.append("<answer>" in text and "</answer>" in text)
            answers.append(gen_num)

        n_correct = sum(correct_flags)
        n_total = len(correct_flags)
        is_mixed = 0 < n_correct < n_total

        problem_id = str(hash(prompt))
        difficulty_weight = (
            self.difficulty.get_difficulty(problem_id)
            if self.config.difficulty_aware else 1.0
        )

        novelty_values = [0.0] * n_total

        if is_mixed and n_correct >= 2:
            correct_indices = [i for i, c in enumerate(correct_flags) if c]
            correct_texts = [texts[i] for i in correct_indices]
            embeddings = self.pse.encode(correct_texts)
            correct_novelty = self._compute_novelty_intra_batch(embeddings)
            for idx, nov in zip(correct_indices, correct_novelty.tolist()):
                novelty_values[idx] = nov
        elif is_mixed and n_correct == 1:
            for i, c in enumerate(correct_flags):
                if c:
                    novelty_values[i] = 1.0

        rewards = []
        for i, (is_correct, has_format) in enumerate(
            zip(correct_flags, format_flags)
        ):
            if is_correct:
                r = 1.0
                if has_format:
                    r += 0.2
                if is_mixed:
                    r += self.config.ss_bonus * novelty_values[i] * difficulty_weight
            else:
                r = 0.0
                if has_format:
                    r += 0.2
                if is_mixed and self.config.err_bonus > 0:
                    ans_nov = self._answer_novelty(answers[i])
                    r += self.config.err_bonus * ans_nov
            rewards.append(r)

        batch_accuracy = n_correct / max(n_total, 1)
        self.difficulty.update(problem_id, batch_accuracy)
        for ans in answers:
            if ans is not None:
                self.answer_counts[ans] += 1
                self.total_answers += 1

        return rewards

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truth: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        if ground_truth is None:
            rewards = []
            for completion in completions:
                text = _completion_to_text(completion)
                fmt = 0.2 if "<answer>" in text and "</answer>" in text else 0.0
                rewards.append(fmt)
            return rewards

        # Group by prompt and process each group independently
        groups = {}
        for i, (prompt, completion, gt) in enumerate(
            zip(prompts, completions, ground_truth)
        ):
            prompt_key = prompt if isinstance(prompt, str) else str(prompt)
            if prompt_key not in groups:
                groups[prompt_key] = []
            groups[prompt_key].append((i, completion, gt))

        all_rewards = [0.0] * len(prompts)
        total_mixed = 0
        total_groups = 0
        total_correct = 0
        total_n = 0

        for prompt_key, group_items in groups.items():
            total_groups += 1
            indices = [item[0] for item in group_items]
            group_completions = [item[1] for item in group_items]
            group_gts = [item[2] for item in group_items]

            group_rewards = self._process_group(
                prompt_key, group_completions, group_gts
            )

            for idx, reward in zip(indices, group_rewards):
                all_rewards[idx] = reward

            n_corr = sum(1 for r in group_rewards if r >= 1.0)
            total_correct += n_corr
            total_n += len(group_items)
            if 0 < n_corr < len(group_items):
                total_mixed += 1

        self._last_metrics = {
            "sus/correctness_mean": total_correct / max(total_n, 1),
            "sus/mixed_batch_frac": total_mixed / max(total_groups, 1),
            "sus/num_groups": float(total_groups),
            "sus/reward_mean": sum(all_rewards) / max(len(all_rewards), 1),
        }

        return all_rewards

    @property
    def last_metrics(self) -> Dict[str, float]:
        return dict(self._last_metrics)


def build_sus_reward(
    ss_bonus: float = 0.2,
    err_bonus: float = 0.0,
    difficulty_aware: bool = True,
    device: str = "cuda",
) -> SuSReward:
    config = SuSConfig(
        ss_bonus=ss_bonus, err_bonus=err_bonus, difficulty_aware=difficulty_aware,
    )
    reward_fn = SuSReward(config, device=device)
    reward_fn.__name__ = "sus_reward"
    return reward_fn


