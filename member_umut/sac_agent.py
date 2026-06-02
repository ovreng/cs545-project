"""
Discrete Soft Actor-Critic (Christodoulou 2019) for BrainBlock.

Architecture:
  - Actor:  encoder → 320 logits → masked softmax → π(a|s)
  - Critics: two separate encoders → 320 Q-values each (twin critics)
  - Target critics: EMA copy of critics (no grad)
  - Temperature α: auto-tuned via gradient descent on log_alpha

Bellman target (discrete SAC):
  V(s') = Σ_a π(a|s') * [min(Q1_tgt, Q2_tgt)(s', a') - α * log π(a|s')]
  y = r + γ * (1 - done) * V(s')

Policy loss:
  J(π) = Σ_a π(a|s) * [α * log π(a|s) - min(Q1, Q2)(s, a)]

Temperature loss:
  J(α) = -α * E_s[Σ_a π(a|s) * log π(a|s) + H_target]
  H_target = target_entropy_ratio * log(320)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from member_umut.network import MLPEncoder, CNNMLPEncoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SACConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005               # soft target update rate

    encoder_type: str = "mlp"
    hidden_dim: int = 256

    buffer_size: int = 500_000
    batch_size: int = 256

    auto_alpha: bool = True
    init_alpha: float = 0.1
    target_entropy_ratio: float = 0.3    # fraction of log(320)≈5.77; ~1.73 nats
                                         # late-episode states have 2-5 valid actions
                                         # → max achievable ≈ log(5)=1.6 nats, so keep low

    learning_starts: int = 10_000


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Preallocated circular replay buffer for off-policy training."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.grids      = np.zeros((capacity, 1, 5, 8), dtype=np.float32)
        self.vecs       = np.zeros((capacity, 10),       dtype=np.float32)
        self.masks      = np.zeros((capacity, 320),      dtype=np.float32)
        self.actions    = np.zeros(capacity,             dtype=np.int64)
        self.rewards    = np.zeros(capacity,             dtype=np.float32)
        self.next_grids = np.zeros((capacity, 1, 5, 8), dtype=np.float32)
        self.next_vecs  = np.zeros((capacity, 10),       dtype=np.float32)
        self.next_masks = np.zeros((capacity, 320),      dtype=np.float32)
        self.dones      = np.zeros(capacity,             dtype=np.float32)

    def add(self, obs: dict, action: int, reward: float,
            next_obs: dict, done: bool,
            action_mask: np.ndarray, next_action_mask: np.ndarray):
        i = self.pos
        self.grids[i]      = obs["grid"]
        self.vecs[i]       = obs["vec"]
        self.masks[i]      = action_mask
        self.actions[i]    = action
        self.rewards[i]    = reward
        self.next_grids[i] = next_obs["grid"]
        self.next_vecs[i]  = next_obs["vec"]
        self.next_masks[i] = next_action_mask
        self.dones[i]      = float(done)

        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        def t(arr): return torch.from_numpy(arr[idx]).to(self.device)
        return {
            "grid":       t(self.grids),
            "vec":        t(self.vecs),
            "mask":       t(self.masks),
            "action":     torch.from_numpy(self.actions[idx]).long().to(self.device),
            "reward":     t(self.rewards),
            "next_grid":  t(self.next_grids),
            "next_vec":   t(self.next_vecs),
            "next_mask":  t(self.next_masks),
            "done":       t(self.dones),
        }

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class SACActorNetwork(nn.Module):
    """Encoder → masked softmax policy over 320 actions."""

    def __init__(self, encoder_type: str = "mlp", hidden_dim: int = 256):
        super().__init__()
        if encoder_type == "mlp":
            self.encoder = MLPEncoder(hidden_dim)
        elif encoder_type == "cnn_mlp":
            self.encoder = CNNMLPEncoder(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        enc_dim = self.encoder.output_dim
        self.actor_head = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 320),
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor,
                action_mask: torch.Tensor):
        """
        Returns:
            probs:     (B, 320) masked action probabilities
            log_probs: (B, 320) log of probs; ~-18 for illegal actions (not -inf)
        """
        features = self.encoder(grid, vec)
        logits = self.actor_head(features)                # (B, 320)
        logits = logits + (action_mask - 1.0) * 1e9      # illegal → -1e9
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)               # stable; ~-18 for masked
        return probs, log_probs

    def get_action(self, grid: torch.Tensor, vec: torch.Tensor,
                   action_mask: torch.Tensor,
                   deterministic: bool = False) -> tuple:
        probs, log_probs = self.forward(grid, vec, action_mask)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action, probs, log_probs


class SACCriticNetwork(nn.Module):
    """Encoder → 320 Q-values (one per action)."""

    def __init__(self, encoder_type: str = "mlp", hidden_dim: int = 256):
        super().__init__()
        if encoder_type == "mlp":
            self.encoder = MLPEncoder(hidden_dim)
        elif encoder_type == "cnn_mlp":
            self.encoder = CNNMLPEncoder(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        enc_dim = self.encoder.output_dim
        self.q_head = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 320),
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Returns Q(s, ·) for all 320 actions: (B, 320)."""
        return self.q_head(self.encoder(grid, vec))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """Discrete SAC agent (Christodoulou 2019)."""

    def __init__(self, config: SACConfig, device: torch.device):
        self.config = config
        self.device = device

        # Actor + twin critics + target critics
        self.actor    = SACActorNetwork(config.encoder_type, config.hidden_dim).to(device)
        self.critic1  = SACCriticNetwork(config.encoder_type, config.hidden_dim).to(device)
        self.critic2  = SACCriticNetwork(config.encoder_type, config.hidden_dim).to(device)
        self.critic1_target = SACCriticNetwork(config.encoder_type, config.hidden_dim).to(device)
        self.critic2_target = SACCriticNetwork(config.encoder_type, config.hidden_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in list(self.critic1_target.parameters()) + list(self.critic2_target.parameters()):
            p.requires_grad = False

        # Optimizers
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),   lr=config.lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=config.lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=config.lr_critic)

        # Target entropy with action masking:
        # Effective action space is much smaller than 320 (typically 50-150 valid
        # actions per step).  Using 0.98*log(320)≈5.65 is unreachable → alpha
        # diverges.  We use ratio * log(320) but callers should pass a low ratio
        # (default 0.5) so the target sits below the achievable masked entropy.
        self.target_entropy = config.target_entropy_ratio * np.log(320)
        self.log_alpha = torch.tensor(
            np.log(config.init_alpha), dtype=torch.float32,
            device=device, requires_grad=config.auto_alpha,
        )
        if config.auto_alpha:
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)

        # Replay buffer
        self.buffer = ReplayBuffer(config.buffer_size, device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: dict, action_mask: np.ndarray,
                      deterministic: bool = False) -> int:
        grid = torch.tensor(obs["grid"],            device=self.device).unsqueeze(0)
        vec  = torch.tensor(obs["vec"],             device=self.device).unsqueeze(0)
        mask = torch.tensor(action_mask.astype(np.float32), device=self.device).unsqueeze(0)
        action, _, _ = self.actor.get_action(grid, vec, mask, deterministic=deterministic)
        return action.item()

    def update(self) -> dict:
        """Sample one minibatch and perform one gradient step."""
        cfg   = self.config
        batch = self.buffer.sample(cfg.batch_size)

        grid       = batch["grid"]
        vec        = batch["vec"]
        mask       = batch["mask"]
        actions    = batch["action"]
        rewards    = batch["reward"]
        next_grid  = batch["next_grid"]
        next_vec   = batch["next_vec"]
        next_mask  = batch["next_mask"]
        dones      = batch["done"]

        # ── Bellman target (no grad) ───────────────────────────────
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_grid, next_vec, next_mask)

            q1_next = self.critic1_target(next_grid, next_vec)   # (B, 320)
            q2_next = self.critic2_target(next_grid, next_vec)
            min_q_next = torch.min(q1_next, q2_next)

            # V(s') = Σ_a π(a|s') * [min_Q(s',a') - α * log π(a|s')]
            # π≈0 for masked actions → contribution ≈ 0 (numerically safe)
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=-1)

            target_q = rewards + cfg.gamma * (1.0 - dones) * v_next   # (B,)

        # ── Critic losses ──────────────────────────────────────────
        q1 = self.critic1(grid, vec).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = self.critic2(grid, vec).gather(1, actions.unsqueeze(1)).squeeze(1)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # ── Actor loss ─────────────────────────────────────────────
        probs, log_probs = self.actor(grid, vec, mask)

        with torch.no_grad():
            min_q_curr = torch.min(
                self.critic1(grid, vec),
                self.critic2(grid, vec),
            )

        # J(π) = Σ_a π(a|s) * [α * log π(a|s) - min_Q(s,a)]
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q_curr)).sum(dim=-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Temperature loss ───────────────────────────────────────
        # Correct form (Christodoulou 2019 / CleanRL):
        #   L(log_α) = -log_α * (E_π[log π(a|s)] + H_target)
        #   ∂L/∂(log_α) = -(E[log π] + H_target) = H(π) - H_target
        # → when H(π) < H_target gradient is negative → optimizer increases log_α ✓
        alpha_loss = torch.tensor(0.0, device=self.device)
        if cfg.auto_alpha:
            with torch.no_grad():
                probs_a, log_probs_a = self.actor(grid, vec, mask)
                # E_π[log π] = -H(π)  (negative number)
                log_pi = (probs_a * log_probs_a).sum(dim=-1)  # (B,)

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            # Clamp log_alpha: α ∈ [exp(-5), exp(1)] ≈ [0.007, 2.7]
            # Without this, alpha diverges when target entropy is unreachable
            # (e.g. late-episode states with only 2-5 valid actions)
            self.log_alpha.data.clamp_(-5.0, 1.0)

        # ── Soft target update ─────────────────────────────────────
        _soft_update(self.critic1, self.critic1_target, cfg.tau)
        _soft_update(self.critic2, self.critic2_target, cfg.tau)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss":   actor_loss.item(),
            "alpha_loss":   alpha_loss.item(),
            "alpha":        self.alpha.item(),
            "q1_mean":      q1.mean().item(),
            "q2_mean":      q2.mean().item(),
            "v_next_mean":  v_next.mean().item(),
        }

    def save(self, path: str):
        torch.save({
            "actor":           self.actor.state_dict(),
            "critic1":         self.critic1.state_dict(),
            "critic2":         self.critic2.state_dict(),
            "critic1_target":  self.critic1_target.state_dict(),
            "critic2_target":  self.critic2_target.state_dict(),
            "log_alpha":       self.log_alpha.data,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])


def _soft_update(source: nn.Module, target: nn.Module, tau: float):
    for src_p, tgt_p in zip(source.parameters(), target.parameters()):
        tgt_p.data.mul_(1.0 - tau).add_(tau * src_p.data)
