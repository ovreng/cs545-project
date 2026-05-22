"""
Training script for BrainBlock Discrete SAC.

Usage:
  python -m member_a.sac_train --reward shaped --encoder mlp --seed 42

Key differences from PPO train.py:
  - Off-policy: replay buffer (500K) instead of rollout buffer
  - Random valid actions during warmup (--learning-starts steps)
  - One gradient step per environment step by default
  - Auto-tuning temperature α (disable with --no-auto-alpha)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from member_a.sac_agent import SACAgent, SACConfig
from member_a.environment import BrainBlockEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train Discrete SAC on BrainBlock")
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["sparse", "shaped"])
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "cnn_mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Actor + critic learning rate (lower = more stable Q-learning)")
    parser.add_argument("--lr-alpha", type=float, default=3e-4,
                        help="Alpha (temperature) learning rate")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft target update rate")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--learning-starts", type=int, default=10_000,
                        help="Random exploration steps before first gradient update")
    parser.add_argument("--update-every", type=int, default=1,
                        help="Perform a gradient step every N environment steps")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Gradient steps per update event")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--init-alpha", type=float, default=0.1,
                        help="Initial temperature α")
    parser.add_argument("--no-auto-alpha", action="store_true",
                        help="Disable automatic temperature tuning")
    parser.add_argument("--target-entropy-ratio", type=float, default=0.3,
                        help="Target entropy as fraction of log(320)≈5.77. Keep low with masking: "
                             "late-episode states have 2-5 valid actions → max ≈log(5)=1.6 nats.")
    parser.add_argument("--no-masking", action="store_true",
                        help="Disable action masking (failure-mode analysis)")
    parser.add_argument("--diversity-bonus", type=float, default=0.0,
                        help="Reward bonus for first-ever tiling (0 = off)")
    parser.add_argument("--log-interval", type=int, default=10_000,
                        help="Print stats every N timesteps")
    parser.add_argument("--save-interval", type=int, default=50_000,
                        help="Save checkpoint every N timesteps")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def make_output_dir(args) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        mask_tag = "nomask" if args.no_masking else "mask"
        div_tag = (f"_div{args.diversity_bonus:.2f}".rstrip("0").rstrip(".")
                   if args.diversity_bonus > 0 else "")
        name = f"sac_{args.encoder}_{args.reward}_{mask_tag}_seed{args.seed}{div_tag}"
        out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def train(args):
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # flush after every newline even when piped

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    config = SACConfig(
        lr_actor=args.lr,
        lr_critic=args.lr,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        encoder_type=args.encoder,
        hidden_dim=args.hidden_dim,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        auto_alpha=not args.no_auto_alpha,
        init_alpha=args.init_alpha,
        target_entropy_ratio=args.target_entropy_ratio,
        learning_starts=args.learning_starts,
    )

    out_dir = make_output_dir(args)
    print(f"Output dir: {out_dir}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    env   = BrainBlockEnv(reward_mode=args.reward)
    agent = SACAgent(config, device)

    actor_params  = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic1.parameters())
    print(f"Actor params: {actor_params:,} | Critic params (×2+target): {critic_params:,}")

    # CSV logging
    csv_path   = out_dir / "metrics.csv"
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestep", "episode",
        "mean_reward", "mean_length", "success_rate", "mean_coverage", "invalid_rate",
        "critic1_loss", "critic2_loss", "actor_loss", "alpha_loss",
        "alpha", "q1_mean", "unique_tilings", "wall_time",
    ])

    # Training state
    global_step   = 0
    episode_count = 0
    start_time    = time.time()

    seen_tilings:      set  = set()
    discovered_tilings: list = []

    best_success_rate = 0.0

    ep_rewards   = []
    ep_lengths   = []
    ep_successes = []
    ep_coverages = []
    ep_invalids  = []

    last_stats: dict = {}

    obs, info     = env.reset(seed=args.seed)
    action_mask   = info["action_mask"]
    ep_reward     = 0.0
    ep_length     = 0

    print(f"Training for {args.total_timesteps:,} timesteps")
    print(f"Warmup: {args.learning_starts:,} random steps | Batch: {args.batch_size}")
    print("=" * 70)

    while global_step < args.total_timesteps:

        # ── Select action ──────────────────────────────────────────
        if global_step < args.learning_starts:
            if global_step % 2000 == 0:
                print(f"  [warmup] {global_step:>6,} / {args.learning_starts:,} steps"
                      f" | buffer: {len(agent.buffer):,}", flush=True)
            valid = np.where(action_mask)[0]
            action = int(np.random.choice(valid))
        else:
            eff_mask = np.ones(320, dtype=np.int8) if args.no_masking else action_mask
            action = agent.select_action(obs, eff_mask)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated

        # ── Tiling tracking + optional diversity bonus ─────────────
        if terminated and next_info.get("termination_reason") == "success":
            tiling_key = frozenset(
                (pt, frozenset(cells)) for pt, cells in env._placed
            )
            if tiling_key not in seen_tilings:
                seen_tilings.add(tiling_key)
                discovered_tilings.append({
                    "tiling_id":   len(discovered_tilings) + 1,
                    "episode":     episode_count,
                    "global_step": global_step,
                    "placed": [
                        {"piece": pt, "cells": sorted(list(cells))}
                        for pt, cells in env._placed
                    ],
                })
                if args.diversity_bonus > 0:
                    reward += args.diversity_bonus

        # ── Store transition ───────────────────────────────────────
        next_action_mask = next_info.get("action_mask", np.ones(320, dtype=np.int8))
        eff_mask      = (np.ones(320, dtype=np.float32) if args.no_masking
                         else action_mask.astype(np.float32))
        eff_next_mask = (np.ones(320, dtype=np.float32) if args.no_masking
                         else next_action_mask.astype(np.float32))

        agent.buffer.add(obs, action, reward, next_obs, done, eff_mask, eff_next_mask)

        ep_reward  += reward
        ep_length  += 1
        global_step += 1

        # ── Episode bookkeeping ────────────────────────────────────
        if done:
            reason = next_info.get("termination_reason", "unknown")
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_length)
            ep_successes.append(1.0 if reason == "success" else 0.0)
            ep_coverages.append(next_info.get("coverage", 0.0))
            ep_invalids.append(1.0 if reason == "illegal_action" else 0.0)
            episode_count += 1

            obs, info   = env.reset()
            action_mask = info["action_mask"]
            ep_reward   = 0.0
            ep_length   = 0
        else:
            obs         = next_obs
            action_mask = next_info["action_mask"]

        # ── Gradient update ────────────────────────────────────────
        if (global_step >= args.learning_starts
                and global_step % args.update_every == 0
                and len(agent.buffer) >= config.batch_size):
            for _ in range(args.gradient_steps):
                last_stats = agent.update()

        # ── Logging ────────────────────────────────────────────────
        if global_step % args.log_interval == 0 and len(ep_rewards) > 0:
            w = min(100, len(ep_rewards))
            mean_reward   = np.mean(ep_rewards[-w:])
            mean_length   = np.mean(ep_lengths[-w:])
            success_rate  = np.mean(ep_successes[-w:])
            mean_coverage = np.mean(ep_coverages[-w:])
            invalid_rate  = np.mean(ep_invalids[-w:])
            wall_time     = time.time() - start_time

            cl1   = last_stats.get("critic1_loss", 0.0)
            cl2   = last_stats.get("critic2_loss", 0.0)
            al    = last_stats.get("actor_loss",   0.0)
            aal   = last_stats.get("alpha_loss",   0.0)
            alpha = last_stats.get("alpha", config.init_alpha)
            q1m   = last_stats.get("q1_mean",      0.0)

            csv_writer.writerow([
                global_step, episode_count,
                f"{mean_reward:.4f}", f"{mean_length:.2f}",
                f"{success_rate:.4f}", f"{mean_coverage:.4f}", f"{invalid_rate:.4f}",
                f"{cl1:.6f}", f"{cl2:.6f}", f"{al:.6f}", f"{aal:.6f}",
                f"{alpha:.6f}", f"{q1m:.4f}",
                len(seen_tilings), f"{wall_time:.1f}",
            ])
            csv_file.flush()

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                agent.save(str(out_dir / "best_model.pt"))

            print(
                f"Step {global_step:>8,} | Ep {episode_count:>6,} | "
                f"R={mean_reward:+.3f} | Len={mean_length:.1f} | "
                f"Succ={success_rate:.3f} [best={best_success_rate:.3f}] | "
                f"Cov={mean_coverage:.3f} | Inv={invalid_rate:.3f} | "
                f"CL={cl1:.4f}/{cl2:.4f} | AL={al:.4f} | "
                f"α={alpha:.4f} | UniqueT={len(seen_tilings):3d} | "
                f"t={wall_time:.0f}s"
            )

        # ── Checkpoint ─────────────────────────────────────────────
        if global_step % args.save_interval == 0:
            ckpt = out_dir / f"checkpoint_{global_step}.pt"
            agent.save(str(ckpt))
            print(f"  [ckpt] {ckpt}")

    # ── Final save ─────────────────────────────────────────────────
    agent.save(str(out_dir / "final_model.pt"))
    csv_file.close()

    with open(out_dir / "discovered_tilings.json", "w") as f:
        json.dump(discovered_tilings, f, indent=2)

    np.savez(
        out_dir / "episode_data.npz",
        rewards=np.array(ep_rewards),
        lengths=np.array(ep_lengths),
        successes=np.array(ep_successes),
        coverages=np.array(ep_coverages),
        invalids=np.array(ep_invalids),
    )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Training complete: {episode_count:,} episodes, {global_step:,} steps in {elapsed:.1f}s")
    print(f"Final success rate: {np.mean(ep_successes[-100:]):.4f}")
    print(f"Unique tilings discovered: {len(seen_tilings)}")
    print(f"Model saved to: {out_dir / 'final_model.pt'}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
