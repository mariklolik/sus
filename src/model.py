import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class TSCLOutput:
    policy_loss: torch.Tensor
    intrinsic_loss: torch.Tensor
    total_loss: torch.Tensor
    strategy_surprise: float = 0.0
    success_surprise: float = 0.0
    mean_pred_success: float = 0.0
    embedding_metrics: Optional[Dict] = None
    diversity_metrics: Optional[Dict] = None


class StrategyPredictionHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        strategy_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.strategy_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, strategy_dim),
        )
        self.success_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_pred = F.normalize(self.strategy_predictor(hidden_state), dim=-1)
        p_success = self.success_predictor(hidden_state).squeeze(-1)
        return s_pred, p_success


class PostHocStrategyEncoder(nn.Module):

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        strategy_dim: int = 128,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.encoder_dim = self.encoder.get_sentence_embedding_dimension()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, strategy_dim),
            nn.GELU(),
            nn.Linear(strategy_dim, strategy_dim),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        embeddings = embeddings.to(self.projector[0].weight.device).to(self.projector[0].weight.dtype)
        projected = self.projector(embeddings)
        return F.normalize(projected, dim=-1)


class TemporalContrastiveReward(nn.Module):

    def __init__(
        self,
        lambda_strategy: float = 0.5,
        lambda_success: float = 0.5,
        only_reward_correct: bool = True,
    ):
        super().__init__()
        self.lambda_strategy = lambda_strategy
        self.lambda_success = lambda_success
        self.only_reward_correct = only_reward_correct

    def forward(
        self,
        s_predicted: torch.Tensor,
        s_actual: torch.Tensor,
        p_predicted_success: torch.Tensor,
        correctness: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        strategy_surprise = 1 - F.cosine_similarity(s_predicted.float(), s_actual.float(), dim=-1)
        success_surprise = torch.abs(correctness.float() - p_predicted_success.float())

        raw_intrinsic = (
            self.lambda_strategy * strategy_surprise +
            self.lambda_success * success_surprise
        )

        if self.only_reward_correct:
            intrinsic_reward = raw_intrinsic * correctness.float()
        else:
            intrinsic_reward = raw_intrinsic

        metrics = {
            "strategy_surprise_mean": strategy_surprise.mean().item(),
            "strategy_surprise_std": strategy_surprise.std().item(),
            "success_surprise_mean": success_surprise.mean().item(),
            "success_surprise_std": success_surprise.std().item(),
            "intrinsic_reward_mean": intrinsic_reward.mean().item(),
            "high_ss_correct_ratio": (
                (strategy_surprise > 0.5) & (correctness == 1)
            ).float().mean().item(),
        }

        return intrinsic_reward, metrics


class GRPOLoss(nn.Module):

    def __init__(self, kl_coef: float = 0.1, clip_range: float = 0.2):
        super().__init__()
        self.kl_coef = kl_coef
        self.clip_range = clip_range

    def forward(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if advantages is None:
            advantages = rewards - rewards.mean()
            std = advantages.std()
            if std > 1e-8:
                advantages = advantages / std

        ratio = torch.exp(logprobs - ref_logprobs)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        kl_div = (ref_logprobs - logprobs).mean()

        total_loss = pg_loss + self.kl_coef * kl_div

        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "mean_ratio": ratio.mean().item(),
        }

        return total_loss, metrics


class SPHTrainer:

    def __init__(
        self,
        sph: StrategyPredictionHead,
        pse: PostHocStrategyEncoder,
        device: str = "cuda",
    ):
        self.sph = sph
        self.pse = pse
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def train_step(
        self,
        hidden_states: torch.Tensor,
        generated_texts: List[str],
        correctness: torch.Tensor,
    ) -> Dict:
        s_pred, p_success = self.sph(hidden_states)

        with torch.no_grad():
            s_actual = self.pse(generated_texts).to(self.device)

        strategy_loss = self.mse_loss(s_pred.float(), s_actual.float())
        success_loss = self.bce_loss(p_success.float(), correctness.float())

        total_loss = strategy_loss + success_loss

        return {
            "total_loss": total_loss,
            "strategy_loss": strategy_loss.item(),
            "success_loss": success_loss.item(),
        }


class TSCLTrainer:

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        hidden_dim: int,
        config: Dict,
        device: str = "cuda",
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.device = device

        tscl_config = config.get("tscl", {})
        grpo_config = config.get("grpo", {})

        strategy_dim = tscl_config.get("strategy_dim", 128)

        self.sph = StrategyPredictionHead(
            hidden_dim=hidden_dim,
            strategy_dim=strategy_dim,
            dropout=tscl_config.get("dropout", 0.1),
        ).to(device).to(torch.bfloat16)

        self.pse = PostHocStrategyEncoder(
            model_name=tscl_config.get("pse_model", "all-MiniLM-L6-v2"),
            strategy_dim=strategy_dim,
            freeze_encoder=tscl_config.get("freeze_pse", True),
        ).to(device).to(torch.bfloat16)

        self.reward_module = TemporalContrastiveReward(
            lambda_strategy=tscl_config.get("lambda_strategy", 0.5),
            lambda_success=tscl_config.get("lambda_success", 0.5),
            only_reward_correct=tscl_config.get("only_reward_correct", True),
        )

        self.grpo_loss_fn = GRPOLoss(
            kl_coef=grpo_config.get("kl_coef", 0.1),
            clip_range=grpo_config.get("clip_range", 0.2),
        )

        self.alpha = tscl_config.get("alpha", 0.3)
        self.tscl_enabled = tscl_config.get("enabled", True)

        self.sph_trainer = SPHTrainer(self.sph, self.pse, device)

        self.trajectory_buffer: List[Dict] = []

    def get_trainable_parameters(self):
        params = list(self.model.parameters())
        params += list(self.sph.parameters())
        params += list(self.pse.projector.parameters())
        return params

    def compute_diversity_metrics(
        self,
        strategy_embeddings: torch.Tensor,
        correctness: torch.Tensor,
    ) -> Dict:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        emb_np = strategy_embeddings.detach().float().cpu().numpy()
        correct_np = correctness.detach().cpu().numpy()

        if len(emb_np) < 4:
            return {
                "strategy_cluster_entropy": 0.0,
                "unique_strategy_ratio": 0.0,
                "correct_diversity": 0.0,
            }

        n_clusters = min(4, len(emb_np) // 2)
        if n_clusters < 2:
            return {
                "strategy_cluster_entropy": 0.0,
                "unique_strategy_ratio": 0.0,
                "correct_diversity": 0.0,
            }

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        cluster_labels = kmeans.fit_predict(emb_np)

        cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
        probs = cluster_counts / cluster_counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        unique_clusters = len(np.unique(cluster_labels))
        unique_ratio = unique_clusters / n_clusters

        correct_mask = correct_np == 1
        if correct_mask.sum() >= 2:
            correct_clusters = cluster_labels[correct_mask]
            correct_unique = len(np.unique(correct_clusters))
            correct_diversity = correct_unique / n_clusters
        else:
            correct_diversity = 0.0

        try:
            silhouette = silhouette_score(emb_np, cluster_labels)
        except Exception:
            silhouette = 0.0

        return {
            "strategy_cluster_entropy": float(entropy),
            "unique_strategy_ratio": float(unique_ratio),
            "correct_diversity": float(correct_diversity),
            "silhouette_score": float(silhouette),
            "num_clusters": int(unique_clusters),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_texts: List[str],
        correctness: torch.Tensor,
        rewards: torch.Tensor,
        query_hidden_states: Optional[torch.Tensor] = None,
        compute_diversity: bool = False,
    ) -> TSCLOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits
        ref_logits = ref_outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()

        logprobs = F.log_softmax(shift_logits, dim=-1)
        logprobs = torch.gather(logprobs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        logprobs = (logprobs * attention_mask[..., 1:]).sum(dim=-1)

        ref_logprobs = F.log_softmax(shift_ref_logits, dim=-1)
        ref_logprobs = torch.gather(ref_logprobs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        ref_logprobs = (ref_logprobs * attention_mask[..., 1:]).sum(dim=-1)

        if query_hidden_states is None:
            query_hidden_states = outputs.hidden_states[-1][:, 0, :]

        s_predicted, p_predicted_success = self.sph(query_hidden_states)

        with torch.no_grad():
            s_actual = self.pse(generated_texts)

        if self.tscl_enabled:
            intrinsic_reward, reward_metrics = self.reward_module(
                s_predicted=s_predicted,
                s_actual=s_actual,
                p_predicted_success=p_predicted_success,
                correctness=correctness,
            )
            combined_rewards = rewards + self.alpha * intrinsic_reward
        else:
            combined_rewards = rewards
            reward_metrics = {
                "strategy_surprise_mean": 0.0,
                "success_surprise_mean": 0.0,
                "intrinsic_reward_mean": 0.0,
            }
            intrinsic_reward = torch.zeros_like(rewards)

        grpo_loss, grpo_metrics = self.grpo_loss_fn(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=combined_rewards,
        )

        if self.tscl_enabled:
            sph_metrics = self.sph_trainer.train_step(
                hidden_states=query_hidden_states,
                generated_texts=generated_texts,
                correctness=correctness,
            )
            intrinsic_loss = sph_metrics["total_loss"]
        else:
            intrinsic_loss = torch.tensor(0.0, device=self.device)

        total_loss = grpo_loss + 0.1 * intrinsic_loss

        diversity_metrics = None
        if compute_diversity:
            diversity_metrics = self.compute_diversity_metrics(s_actual, correctness)

        return TSCLOutput(
            policy_loss=grpo_loss,
            intrinsic_loss=intrinsic_loss,
            total_loss=total_loss,
            strategy_surprise=reward_metrics.get("strategy_surprise_mean", 0.0),
            success_surprise=reward_metrics.get("success_surprise_mean", 0.0),
            mean_pred_success=p_predicted_success.mean().item(),
            embedding_metrics=reward_metrics,
            diversity_metrics=diversity_metrics,
        )


class BaselineTrainer:

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: Dict,
        device: str = "cuda",
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.device = device

        grpo_config = config.get("grpo", {})
        self.grpo_loss_fn = GRPOLoss(
            kl_coef=grpo_config.get("kl_coef", 0.1),
            clip_range=grpo_config.get("clip_range", 0.2),
        )

    def get_trainable_parameters(self):
        return list(self.model.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits
        ref_logits = ref_outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()

        logprobs = F.log_softmax(shift_logits, dim=-1)
        logprobs = torch.gather(logprobs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        logprobs = (logprobs * attention_mask[..., 1:]).sum(dim=-1)

        ref_logprobs = F.log_softmax(shift_ref_logits, dim=-1)
        ref_logprobs = torch.gather(ref_logprobs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        ref_logprobs = (ref_logprobs * attention_mask[..., 1:]).sum(dim=-1)

        loss, metrics = self.grpo_loss_fn(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=rewards,
        )

        return loss, metrics


class PerplexityRewardTrainer:

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: Dict,
        device: str = "cuda",
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.device = device

        grpo_config = config.get("grpo", {})
        cde_config = config.get("cde", {})

        self.grpo_loss_fn = GRPOLoss(
            kl_coef=grpo_config.get("kl_coef", 0.1),
            clip_range=grpo_config.get("clip_range", 0.2),
        )

        self.beta = cde_config.get("beta", 0.1)
        self.perplexity_scale = cde_config.get("perplexity_scale", 0.01)

    def get_trainable_parameters(self):
        return list(self.model.parameters())

    def compute_perplexity_reward(
        self,
        logprobs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_lens = attention_mask[:, 1:].sum(dim=-1)
        mean_logprob = logprobs / (seq_lens + 1e-8)
        perplexity = torch.exp(-mean_logprob)
        perplexity_reward = self.perplexity_scale * perplexity
        return perplexity_reward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        correctness: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits
        ref_logits = ref_outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()

        logprobs = F.log_softmax(shift_logits, dim=-1)
        logprobs = torch.gather(logprobs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        logprobs = (logprobs * attention_mask[..., 1:]).sum(dim=-1)

        ref_logprobs = F.log_softmax(shift_ref_logits, dim=-1)
        ref_logprobs = torch.gather(ref_logprobs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        ref_logprobs = (ref_logprobs * attention_mask[..., 1:]).sum(dim=-1)

        perplexity_reward = self.compute_perplexity_reward(logprobs, attention_mask)
        perplexity_reward = perplexity_reward * correctness.float()

        combined_rewards = rewards + self.beta * perplexity_reward

        loss, metrics = self.grpo_loss_fn(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=combined_rewards,
        )

        metrics["perplexity_reward_mean"] = perplexity_reward.mean().item()

        return loss, metrics
