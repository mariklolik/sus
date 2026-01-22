"""
SuS: Strategy-aware Surprise for Intrinsic Exploration

Implementation aligned with arXiv:2601.10349

Key Components:
- Strategy Stability (SS): Temporal consistency measure (Equation 2)
- Strategy Surprise (SuS): World model prediction error weighted by instability (Equation 3)
- World Model M: Predicts next state from current state and action
- InfoNCE Contrastive Learning: For stable strategy representations
- Momentum Encoder: MoCo-style stable target embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import copy


@dataclass
class SuSOutput:
    """Output from SuS trainer forward pass."""
    policy_loss: torch.Tensor
    intrinsic_loss: torch.Tensor
    total_loss: torch.Tensor
    strategy_stability: float = 0.0  # SS metric (renamed from strategy_surprise)
    strategy_surprise: float = 0.0   # SuS metric (renamed from success_surprise)
    mean_pred_success: float = 0.0
    embedding_metrics: Optional[Dict] = None
    diversity_metrics: Optional[Dict] = None


class WorldModel(nn.Module):
    """
    World Model M that predicts next state from current state and action.

    As per paper: M(s_t, a_t) -> s_hat_{t+1}

    Architecture: Three-layer MLP with ReLU activations and layer normalization.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Action embedding layer as described in paper
        self.action_embedding = nn.Linear(action_dim, state_dim)

        # Three-layer MLP with ReLU and LayerNorm as per paper
        self.mlp = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        action_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next state from current state and action.

        Args:
            state: Current state embedding [batch_size, state_dim]
            action_embedding: Action embedding [batch_size, action_dim]

        Returns:
            Predicted next state embedding [batch_size, state_dim]
        """
        action_emb = self.action_embedding(action_embedding)
        combined = torch.cat([state, action_emb], dim=-1)
        predicted_next_state = self.mlp(combined)
        return F.normalize(predicted_next_state, dim=-1)


class StrategyPredictionHead(nn.Module):
    """
    Strategy Prediction Head (SPH) for predicting strategy and success.

    Architecture: Three-layer MLP with ReLU activations and layer normalization
    as specified in the paper.
    """

    def __init__(
        self,
        hidden_dim: int,
        strategy_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Three-layer MLP with ReLU and LayerNorm as per paper
        self.strategy_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, strategy_dim),
        )

        # Success prediction head (separate)
        self.success_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict strategy embedding and success probability.

        Args:
            hidden_state: Hidden state from LLM [batch_size, hidden_dim]

        Returns:
            Tuple of (strategy_embedding, success_probability)
        """
        s_pred = F.normalize(self.strategy_predictor(hidden_state), dim=-1)
        p_success = self.success_predictor(hidden_state).squeeze(-1)
        return s_pred, p_success


class MomentumEncoder(nn.Module):
    """
    Momentum Encoder for stable target embeddings.

    Implements MoCo-style momentum update: theta_k = m * theta_k + (1-m) * theta_q
    as described in the paper.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.momentum = momentum

        # Create momentum encoder as a copy of base encoder
        self.encoder = copy.deepcopy(base_encoder)

        # Freeze momentum encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, base_encoder: nn.Module):
        """
        Update momentum encoder with exponential moving average.

        theta_k = m * theta_k + (1-m) * theta_q
        """
        for param_q, param_k in zip(
            base_encoder.parameters(),
            self.encoder.parameters()
        ):
            param_k.data = (
                self.momentum * param_k.data +
                (1 - self.momentum) * param_q.data
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through momentum encoder (no gradients)."""
        with torch.no_grad():
            return self.encoder(x)


class PostHocStrategyEncoder(nn.Module):
    """
    Post-Hoc Strategy Encoder (PSE) for encoding generated text into strategy embeddings.

    Uses pre-trained SentenceTransformer with a trainable projection head.
    Architecture follows paper: two-layer projection with layer normalization.
    """

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

        # Projection head with LayerNorm as per paper
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, strategy_dim),
            nn.LayerNorm(strategy_dim),
            nn.ReLU(),
            nn.Linear(strategy_dim, strategy_dim),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into strategy embeddings.

        Args:
            texts: List of generated text strings

        Returns:
            Strategy embeddings [batch_size, strategy_dim]
        """
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        # Clone to escape inference mode and enable gradient tracking through projector
        embeddings = embeddings.clone().to(self.projector[0].weight.device).to(self.projector[0].weight.dtype)
        projected = self.projector(embeddings)
        return F.normalize(projected, dim=-1)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss with temperature scaling.

    As described in paper: uses in-batch negatives with temperature-scaled similarities.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            query: Query embeddings [batch_size, dim]
            positive: Positive sample embeddings [batch_size, dim]
            negatives: Optional negative samples [batch_size, num_negatives, dim]
                      If None, uses in-batch negatives

        Returns:
            InfoNCE loss scalar
        """
        batch_size = query.size(0)

        # Positive similarity
        pos_sim = F.cosine_similarity(query, positive, dim=-1) / self.temperature

        if negatives is None:
            # Use in-batch negatives: each sample's positives are negatives for others
            # Compute all pairwise similarities
            sim_matrix = torch.mm(query, positive.t()) / self.temperature

            # Labels: diagonal elements are positives
            labels = torch.arange(batch_size, device=query.device)

            return F.cross_entropy(sim_matrix, labels)
        else:
            # Use provided negatives
            neg_sim = torch.bmm(
                negatives,
                query.unsqueeze(-1)
            ).squeeze(-1) / self.temperature

            # Concatenate positive and negatives
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)

            return F.cross_entropy(logits, labels)


class SuSRewardModule(nn.Module):
    """
    Strategy-aware Surprise Reward Module.

    Implements the intrinsic reward from the paper:
    r_int = lambda_SS * SS + lambda_SuS * SuS (Equation 4)

    Where:
    - SS = cosine_similarity(z_pre, z_post) (Equation 2)
    - SuS = ||s_hat_{t+1} - s_{t+1}|| * (1 - SS) (Equation 3)
    """

    def __init__(
        self,
        lambda_ss: float = 1.0,  # Paper optimal: 1.0
        lambda_sus: float = 0.0,  # Paper optimal: 0.0
        only_reward_correct: bool = True,
    ):
        super().__init__()
        self.lambda_ss = lambda_ss
        self.lambda_sus = lambda_sus
        self.only_reward_correct = only_reward_correct

    def forward(
        self,
        z_pre: torch.Tensor,           # State embedding BEFORE action (query)
        z_post: torch.Tensor,          # State embedding AFTER action (response)
        s_hat_next: torch.Tensor,      # World model prediction of next state
        correctness: torch.Tensor,     # Binary correctness labels
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute intrinsic reward based on Strategy Stability and Strategy Surprise.

        Args:
            z_pre: Pre-action state embedding [batch_size, dim]
            z_post: Post-action state embedding [batch_size, dim]
            s_hat_next: World model predicted next state [batch_size, dim]
            correctness: Binary correctness [batch_size]

        Returns:
            Tuple of (intrinsic_reward, metrics_dict)
        """
        # Strategy Stability (SS) - Equation 2
        # SS = sim(z_pre, z_post) = cosine_similarity
        # High SS = maintained strategic coherence
        strategy_stability = F.cosine_similarity(
            z_pre.float(),
            z_post.float(),
            dim=-1
        )

        # World model prediction error: ||s_hat_{t+1} - s_{t+1}||
        prediction_error = torch.norm(
            s_hat_next.float() - z_post.float(),
            dim=-1
        )

        # Strategy Surprise (SuS) - Equation 3
        # SuS = ||s_hat_{t+1} - s_{t+1}|| * (1 - SS)
        # Prediction error weighted by strategy instability
        strategy_surprise = prediction_error * (1 - strategy_stability)

        # Intrinsic reward - Equation 4
        # r_int = lambda_SS * SS + lambda_SuS * SuS
        raw_intrinsic = (
            self.lambda_ss * strategy_stability +
            self.lambda_sus * strategy_surprise
        )

        if self.only_reward_correct:
            intrinsic_reward = raw_intrinsic * correctness.float()
        else:
            intrinsic_reward = raw_intrinsic

        metrics = {
            "strategy_stability_mean": strategy_stability.mean().item(),
            "strategy_stability_std": strategy_stability.std().item(),
            "strategy_surprise_mean": strategy_surprise.mean().item(),
            "strategy_surprise_std": strategy_surprise.std().item(),
            "prediction_error_mean": prediction_error.mean().item(),
            "intrinsic_reward_mean": intrinsic_reward.mean().item(),
            # High SS (stability) with correct solutions ratio
            "high_ss_correct_ratio": (
                (strategy_stability > 0.5) & (correctness == 1)
            ).float().mean().item(),
        }

        return intrinsic_reward, metrics


class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization Loss.

    PPO-style clipped objective with KL divergence constraint.
    """

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
    """
    Strategy Prediction Head Trainer.

    Trains SPH using InfoNCE contrastive loss (as per paper)
    instead of simple MSE + BCE.
    """

    def __init__(
        self,
        sph: StrategyPredictionHead,
        pse: PostHocStrategyEncoder,
        world_model: WorldModel,
        momentum_encoder: Optional[MomentumEncoder] = None,
        temperature: float = 0.07,
        device: str = "cuda",
        use_contrastive: bool = True,
    ):
        self.sph = sph
        self.pse = pse
        self.world_model = world_model
        self.momentum_encoder = momentum_encoder
        self.device = device
        self.use_contrastive = use_contrastive

        self.info_nce = InfoNCELoss(temperature=temperature)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def train_step(
        self,
        query_hidden_states: torch.Tensor,   # z_pre - state before action
        response_hidden_states: torch.Tensor, # z_post - state after action
        generated_texts: List[str],
        correctness: torch.Tensor,
    ) -> Dict:
        """
        Train SPH and World Model.

        Args:
            query_hidden_states: Hidden states from query (before action)
            response_hidden_states: Hidden states from response (after action)
            generated_texts: Generated text strings
            correctness: Binary correctness labels

        Returns:
            Dictionary of losses and metrics
        """
        # Get strategy predictions from SPH
        s_pred, p_success = self.sph(query_hidden_states)

        # Get actual strategy embeddings from generated text
        with torch.no_grad():
            s_actual = self.pse(generated_texts).to(self.device)

        # Get target embeddings (using momentum encoder if available)
        if self.momentum_encoder is not None:
            with torch.no_grad():
                s_target = self.momentum_encoder(query_hidden_states)
        else:
            s_target = s_actual

        # Strategy loss
        if self.use_contrastive:
            # InfoNCE contrastive loss as per paper
            strategy_loss = self.info_nce(s_pred, s_target)
        else:
            # Fallback to MSE for compatibility
            strategy_loss = self.mse_loss(s_pred.float(), s_actual.float())

        # World model loss: predict next state from current state and action
        # Action is represented by the generated text embedding
        action_embedding = s_actual  # Use text embedding as action representation
        s_hat_next = self.world_model(s_pred, action_embedding)

        # World model prediction target is the actual next state (response embedding)
        # For LLM, we use the PSE encoding of response as the "next state"
        world_model_loss = self.mse_loss(s_hat_next.float(), s_actual.float())

        # Success prediction loss
        success_loss = self.bce_loss(p_success.float(), correctness.float())

        total_loss = strategy_loss + world_model_loss + success_loss

        # Update momentum encoder if present
        if self.momentum_encoder is not None:
            self.momentum_encoder.update(self.sph.strategy_predictor)

        return {
            "total_loss": total_loss,
            "strategy_loss": strategy_loss.item(),
            "world_model_loss": world_model_loss.item(),
            "success_loss": success_loss.item(),
        }


class SuSTrainer:
    """
    Main SuS Trainer implementing the full algorithm from the paper.

    Combines:
    - Strategy Prediction Head (SPH)
    - Post-Hoc Strategy Encoder (PSE)
    - World Model M
    - Momentum Encoder
    - InfoNCE Contrastive Learning
    - GRPO Policy Optimization
    """

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

        sus_config = config.get("tscl", {})  # Keep "tscl" key for backwards compatibility
        grpo_config = config.get("grpo", {})

        strategy_dim = sus_config.get("strategy_dim", 128)

        # Strategy Prediction Head
        self.sph = StrategyPredictionHead(
            hidden_dim=hidden_dim,
            strategy_dim=strategy_dim,
            dropout=sus_config.get("dropout", 0.1),
        ).to(device).to(torch.bfloat16)

        # Post-Hoc Strategy Encoder
        self.pse = PostHocStrategyEncoder(
            model_name=sus_config.get("pse_model", "all-MiniLM-L6-v2"),
            strategy_dim=strategy_dim,
            freeze_encoder=sus_config.get("freeze_pse", True),
        ).to(device).to(torch.bfloat16)

        # World Model (NEW - required by paper)
        self.world_model = WorldModel(
            state_dim=strategy_dim,
            action_dim=strategy_dim,  # Action is text embedding
            hidden_dim=sus_config.get("world_model_hidden_dim", 256),
            dropout=sus_config.get("dropout", 0.1),
        ).to(device).to(torch.bfloat16)

        # Momentum Encoder (NEW - required by paper)
        use_momentum = sus_config.get("use_momentum_encoder", True)
        if use_momentum:
            self.momentum_encoder = MomentumEncoder(
                base_encoder=self.sph.strategy_predictor,
                momentum=sus_config.get("momentum", 0.999),
            ).to(device).to(torch.bfloat16)
        else:
            self.momentum_encoder = None

        # SuS Reward Module (renamed from TemporalContrastiveReward)
        self.reward_module = SuSRewardModule(
            lambda_ss=sus_config.get("lambda_ss", 1.0),      # Paper optimal: 1.0
            lambda_sus=sus_config.get("lambda_sus", 0.0),    # Paper optimal: 0.0
            only_reward_correct=sus_config.get("only_reward_correct", True),
        )

        # GRPO Loss
        self.grpo_loss_fn = GRPOLoss(
            kl_coef=grpo_config.get("kl_coef", 0.1),
            clip_range=grpo_config.get("clip_range", 0.2),
        )

        self.alpha = sus_config.get("alpha", 0.3)
        self.sus_enabled = sus_config.get("enabled", True)

        # SPH Trainer with contrastive learning
        self.sph_trainer = SPHTrainer(
            sph=self.sph,
            pse=self.pse,
            world_model=self.world_model,
            momentum_encoder=self.momentum_encoder,
            temperature=sus_config.get("temperature", 0.07),
            device=device,
            use_contrastive=sus_config.get("use_contrastive", True),
        )

        self.intrinsic_loss_coef = sus_config.get("intrinsic_loss_coef", 0.1)

    def get_trainable_parameters(self):
        params = list(self.model.parameters())
        params += list(self.sph.parameters())
        params += list(self.pse.projector.parameters())
        params += list(self.world_model.parameters())
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
    ) -> SuSOutput:
        """
        Forward pass implementing the full SuS algorithm.

        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask
            generated_texts: List of generated text strings
            correctness: Binary correctness labels
            rewards: External rewards
            query_hidden_states: Pre-computed query hidden states (z_pre)
            compute_diversity: Whether to compute diversity metrics

        Returns:
            SuSOutput with losses and metrics
        """
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

        # z_pre: State before action (query hidden state)
        if query_hidden_states is None:
            query_hidden_states = outputs.hidden_states[-1][:, 0, :]
        z_pre = query_hidden_states

        # Get response hidden states (z_post approximation)
        # Use the last hidden state of the full sequence
        response_hidden_states = outputs.hidden_states[-1][:, -1, :]

        # Predict strategy from query (z_pre)
        s_predicted, p_predicted_success = self.sph(z_pre)

        # z_post: State after action (from generated text)
        with torch.no_grad():
            z_post = self.pse(generated_texts)

        if self.sus_enabled:
            # World model prediction: s_hat_{t+1} = M(z_pre, action)
            # Action is represented by the generated text embedding
            action_embedding = z_post  # Use text embedding as action representation
            s_hat_next = self.world_model(s_predicted, action_embedding)

            # Compute intrinsic reward using correct SS and SuS formulas
            intrinsic_reward, reward_metrics = self.reward_module(
                z_pre=s_predicted,     # Strategy embedding of query
                z_post=z_post,         # Strategy embedding of response
                s_hat_next=s_hat_next, # World model prediction
                correctness=correctness,
            )
            combined_rewards = rewards + self.alpha * intrinsic_reward
        else:
            combined_rewards = rewards
            reward_metrics = {
                "strategy_stability_mean": 0.0,
                "strategy_surprise_mean": 0.0,
                "intrinsic_reward_mean": 0.0,
            }
            intrinsic_reward = torch.zeros_like(rewards)

        grpo_loss, grpo_metrics = self.grpo_loss_fn(
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=combined_rewards,
        )

        if self.sus_enabled:
            sph_metrics = self.sph_trainer.train_step(
                query_hidden_states=z_pre,
                response_hidden_states=response_hidden_states,
                generated_texts=generated_texts,
                correctness=correctness,
            )
            intrinsic_loss = sph_metrics["total_loss"]
        else:
            intrinsic_loss = torch.tensor(0.0, device=self.device)

        total_loss = grpo_loss + self.intrinsic_loss_coef * intrinsic_loss

        diversity_metrics = None
        if compute_diversity:
            diversity_metrics = self.compute_diversity_metrics(z_post, correctness)

        return SuSOutput(
            policy_loss=grpo_loss,
            intrinsic_loss=intrinsic_loss,
            total_loss=total_loss,
            strategy_stability=reward_metrics.get("strategy_stability_mean", 0.0),
            strategy_surprise=reward_metrics.get("strategy_surprise_mean", 0.0),
            mean_pred_success=p_predicted_success.mean().item(),
            embedding_metrics=reward_metrics,
            diversity_metrics=diversity_metrics,
        )


# Backwards compatibility aliases
TSCLOutput = SuSOutput
TSCLTrainer = SuSTrainer
TemporalContrastiveReward = SuSRewardModule


class BaselineTrainer:
    """Baseline trainer using only GRPO without intrinsic rewards."""

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
    """Perplexity-based intrinsic reward trainer (CDE baseline)."""

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
