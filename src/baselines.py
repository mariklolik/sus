#!/usr/bin/env python3
"""Baseline exploration reward functions for GRPO.

Methods:
  1. entropy_bonus     — n-gram diversity bonus (EDGE-GRPO)
  2. rs_grpo           — risk-seeking advantage (arXiv:2509.24261)
  3. progrpo_arm       — progressive advantage re-weighting (arXiv:2602.05281)
  4. ceeh_difficulty   — difficulty-aware entropy regularization
  5. ucb_exploration   — UCB bonus for rare answers
"""

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional

from train_grpo import extract_generated, extract_gsm8k_gt, _numbers_match


def _completion_to_text(completion) -> str:
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    elif isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# --- Method 1: Entropy Bonus ---

def entropy_bonus_reward(
    prompts: List[str],
    completions: List[str],
    *,
    beta: float = 0.3,
    ngram_sizes: tuple = (2, 3),
    **kwargs,
) -> List[float]:
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        tokens = text.split()
        if len(tokens) < 2:
            rewards.append(0.0)
            continue
        ratios = []
        for n in ngram_sizes:
            grams = _ngrams(tokens, n)
            if grams:
                ratios.append(len(set(grams)) / len(grams))
            else:
                ratios.append(0.0)
        diversity_score = sum(ratios) / len(ratios) if ratios else 0.0
        rewards.append(diversity_score * beta)
    return rewards


# --- Method 2: RS-GRPO ---

class RSGRPOReward:
    def __init__(self, alpha: float = 0.5, bonus: float = 0.5):
        self.alpha = alpha
        self.bonus = bonus

    def __call__(
        self, prompts: List[str], completions: List[str],
        ground_truth: Optional[List[str]] = None, **kwargs,
    ) -> List[float]:
        if ground_truth is None:
            return [0.0] * len(completions)
        scores = []
        for completion, gt in zip(completions, ground_truth):
            text = _completion_to_text(completion)
            gt_num = extract_gsm8k_gt(gt)
            gen_num = extract_generated(text)
            if gt_num is not None and gen_num is not None and _numbers_match(gen_num, gt_num):
                scores.append(1.0)
            else:
                scores.append(0.0)
        if not scores:
            return [0.0] * len(completions)
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        threshold = self.alpha * max_score + (1.0 - self.alpha) * mean_score
        return [self.bonus if s >= threshold and s > 0 else 0.0 for s in scores]


# --- Method 3: ProGRPO / ARM ---

class ProGRPOReward:
    def __init__(self, ema_alpha: float = 0.1, bonus_scale: float = 0.5):
        self.ema_alpha = ema_alpha
        self.bonus_scale = bonus_scale
        self.problem_accuracy: Dict[str, float] = {}

    def _update_ema(self, problem_id: str, correct: float):
        if problem_id in self.problem_accuracy:
            old = self.problem_accuracy[problem_id]
            self.problem_accuracy[problem_id] = (
                self.ema_alpha * correct + (1.0 - self.ema_alpha) * old
            )
        else:
            self.problem_accuracy[problem_id] = correct

    def __call__(
        self, prompts: List[str], completions: List[str],
        ground_truth: Optional[List[str]] = None, **kwargs,
    ) -> List[float]:
        if ground_truth is None:
            return [0.0] * len(completions)
        rewards = []
        for prompt, completion, gt in zip(prompts, completions, ground_truth):
            text = _completion_to_text(completion)
            gt_num = extract_gsm8k_gt(gt)
            gen_num = extract_generated(text)
            correct = (
                1.0
                if gt_num is not None and gen_num is not None
                and _numbers_match(gen_num, gt_num)
                else 0.0
            )
            problem_id = str(hash(prompt))
            ema = self.problem_accuracy.get(problem_id, 0.5)
            difficulty_weight = 1.0 - ema
            rewards.append(self.bonus_scale * difficulty_weight if correct > 0 else 0.0)
            self._update_ema(problem_id, correct)
        return rewards


# --- Method 4: Difficulty-Aware (CEEH style) ---

class DifficultyTracker:
    def __init__(self, ema_alpha: float = 0.1, hard_threshold: float = 0.4):
        self.ema_alpha = ema_alpha
        self.hard_threshold = hard_threshold
        self.problem_accuracy: Dict[str, float] = {}

    def update(self, problem_id: str, correct: float):
        if problem_id in self.problem_accuracy:
            old = self.problem_accuracy[problem_id]
            self.problem_accuracy[problem_id] = (
                self.ema_alpha * correct + (1.0 - self.ema_alpha) * old
            )
        else:
            self.problem_accuracy[problem_id] = correct

    def is_hard(self, problem_id: str) -> bool:
        return self.problem_accuracy.get(problem_id, 0.5) < self.hard_threshold


def difficulty_aware_reward(
    prompts: List[str], completions: List[str],
    ground_truth: Optional[List[str]] = None,
    *, diversity_beta: float = 0.3, ngram_sizes: tuple = (2, 3),
    tracker: Optional[DifficultyTracker] = None, **kwargs,
) -> List[float]:
    if tracker is None:
        tracker = DifficultyTracker()
    if ground_truth is None:
        return [0.0] * len(completions)
    rewards = []
    for prompt, completion, gt in zip(prompts, completions, ground_truth):
        text = _completion_to_text(completion)
        problem_id = str(hash(prompt))
        gt_num = extract_gsm8k_gt(gt)
        gen_num = extract_generated(text)
        correct = (
            1.0
            if gt_num is not None and gen_num is not None
            and _numbers_match(gen_num, gt_num)
            else 0.0
        )
        tokens = text.split()
        if len(tokens) >= 2 and tracker.is_hard(problem_id):
            ratios = []
            for n in ngram_sizes:
                grams = _ngrams(tokens, n)
                if grams:
                    ratios.append(len(set(grams)) / len(grams))
            diversity_score = sum(ratios) / len(ratios) if ratios else 0.0
            reward = diversity_score * diversity_beta
        else:
            reward = 0.0
        rewards.append(reward)
        tracker.update(problem_id, correct)
    return rewards


# --- Method 5: UCB Exploration ---

class UCBExplorer:
    def __init__(self, c: float = 0.5):
        self.c = c
        self.answer_counts: Dict[str, int] = defaultdict(int)
        self.total: int = 0

    def ucb_bonus(self, answer: Optional[str]) -> float:
        if answer is None or self.total == 0:
            return self.c
        count = self.answer_counts.get(answer, 0)
        if count == 0:
            return self.c
        return self.c * math.sqrt(math.log(self.total) / count)

    def _record(self, answer: Optional[str]):
        self.total += 1
        if answer is not None:
            self.answer_counts[answer] += 1

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            text = _completion_to_text(completion)
            answer = extract_generated(text)
            bonus = self.ucb_bonus(answer)
            rewards.append(bonus)
            self._record(answer)
        return rewards


# --- Registry ---

def build_baseline(method: str):
    if method == "entropy_bonus":
        return entropy_bonus_reward
    elif method == "rs_grpo":
        fn = RSGRPOReward(alpha=0.5, bonus=0.5)
        fn.__name__ = "rs_grpo_reward"
        return fn
    elif method == "progrpo_arm":
        fn = ProGRPOReward(ema_alpha=0.1, bonus_scale=0.5)
        fn.__name__ = "progrpo_arm_reward"
        return fn
    elif method == "ceeh_difficulty":
        tracker = DifficultyTracker(ema_alpha=0.1, hard_threshold=0.4)
        def _ceeh(prompts, completions, **kwargs):
            return difficulty_aware_reward(
                prompts, completions, tracker=tracker, **kwargs
            )
        return _ceeh
    elif method == "ucb_exploration":
        fn = UCBExplorer(c=0.5)
        fn.__name__ = "ucb_exploration_reward"
        return fn
    else:
        raise ValueError(
            f"Unknown baseline '{method}'. "
            "Choose from: entropy_bonus, rs_grpo, progrpo_arm, ceeh_difficulty, ucb_exploration"
        )
