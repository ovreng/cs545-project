"""
Training script for BrainBlock DQN (Member B).

Hyperparameters aligned with project_plan.md §4:
  lr=3e-4, gamma=0.99, batch_size=64, hidden_dim=256, max_grad_norm=0.5

DQN-specific additions (no PPO params apply):
  tau=0.005 (soft target update), epsilon schedule, replay buffer size=100k

Usage:
  python -m member_b.train --episodes 5000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from member_b.agent import DQNAgent
from member_b.environment import BrainBlockEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on BrainBlock (Member B)")
    # ── Shared §4 hyperparameters ─────────────────────────────────────
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (§4: 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor γ (§4: 0.99)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (§4: 64)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="MLP hidden dim (§4: 256)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm (§4: 0.5)")
    # ── DQN-specific hyperparameters ──────────────────────────────────
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft target-network update coefficient")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Starting epsilon for ε-greedy")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=int, default=2000,
                        help="Episodes over which epsilon decays (half of run budget)")
    parser.add_argument("--replay-size", type=int, default=100_000,
                        help="Replay buffer capacity")
    parser.add_argument("--warmup-episodes", type=int, default=200,
                        help="Episodes of pure random play before learning starts")
    # ── Run settings ──────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Total training episodes")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def make_output_dir(args) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = Path("results_b") / f"dqn_sparse_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"[Seed {args.seed}] Device: {device}")

    out_dir = make_output_dir(args)
    print(f"[Seed {args.seed}] Output: {out_dir}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    env = BrainBlockEnv()
    agent = DQNAgent(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        max_grad_norm=args.max_grad_norm,
        replay_size=args.replay_size,
        device=device,
    )

    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "episode", "mean_reward", "std_reward", "mean_length",
        "success_rate", "mean_coverage", "epsilon", "mean_loss", "wall_time",
    ])

    start_time = time.time()
    ep_rewards, ep_lengths, ep_successes, ep_coverages, all_losses = [], [], [], [], []

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode)
        action_mask = info["action_mask"]
        ep_reward = 0.0
        ep_length = 0

        # ε schedule: exponential decay
        if episode <= args.warmup_episodes:
            eps = 1.0
        else:
            eps = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
                  np.exp(-1. * (episode - args.warmup_episodes) / args.epsilon_decay)

        done = False
        while not done:
            action = agent.act(obs, action_mask, epsilon=eps)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_action_mask = next_info["action_mask"]

            agent.memory.push(obs, action, reward, next_obs, done, next_action_mask)

            # Only update after warmup
            if episode > args.warmup_episodes:
                loss = agent.update(args.batch_size)
                if loss > 0:
                    all_losses.append(loss)

            obs = next_obs
            action_mask = next_action_mask
            ep_reward += reward
            ep_length += 1

        reason = next_info.get("termination_reason", "unknown")
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)
        ep_successes.append(1.0 if reason == "success" else 0.0)
        ep_coverages.append(next_info.get("coverage", 0.0))

        if episode % args.log_interval == 0:
            w = min(100, len(ep_rewards))
            r = ep_rewards[-w:]
            mean_r   = float(np.mean(r))
            std_r    = float(np.std(r))
            mean_len = float(np.mean(ep_lengths[-w:]))
            succ     = float(np.mean(ep_successes[-w:]))
            cov      = float(np.mean(ep_coverages[-w:]))
            mean_loss = float(np.mean(all_losses[-200:])) if all_losses else 0.0
            t        = time.time() - start_time

            writer.writerow([
                episode, f"{mean_r:.4f}", f"{std_r:.4f}", f"{mean_len:.2f}",
                f"{succ:.4f}", f"{cov:.4f}", f"{eps:.4f}",
                f"{mean_loss:.6f}", f"{t:.1f}",
            ])
            csv_file.flush()

            print(
                f"[S{args.seed}] Ep {episode:>5,} | "
                f"R={mean_r:+.2f}±{std_r:.2f} | Len={mean_len:.1f} | "
                f"Succ={succ:.3f} | Cov={cov:.3f} | "
                f"ε={eps:.3f} | Loss={mean_loss:.4f} | t={t:.0f}s"
            )

        if episode % args.save_interval == 0:
            agent.save(str(out_dir / f"checkpoint_{episode}.pt"))

    agent.save(str(out_dir / "final_model.pt"))
    csv_file.close()

    np.savez(
        out_dir / "episode_data.npz",
        rewards=np.array(ep_rewards),
        lengths=np.array(ep_lengths),
        successes=np.array(ep_successes),
        coverages=np.array(ep_coverages),
    )
    elapsed = time.time() - start_time
    final_succ = float(np.mean(ep_successes[-100:]))
    print(f"[Seed {args.seed}] Done — {args.episodes} eps in {elapsed:.0f}s | final succ={final_succ:.4f}")
    return final_succ


if __name__ == "__main__":
    args = parse_args()
    train(args)
