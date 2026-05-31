"""
Evaluation script for BrainBlock Discrete SAC.

Usage:
  python -m member_a.sac_evaluate \
      --model results/sac_mlp_shaped_mask_seed42/final_model.pt \
      --encoder mlp --reward shaped --episodes 1000

Features:
  - Deterministic rollouts (argmax policy) by default
  - --stochastic: sample from policy (shows inherent SAC diversity)
  - Reports success rate, mean return, mean length, invalid-action rate
  - Tracks and saves distinct solutions (unique tilings)
  - Renders solution boards and optional step-by-step trace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from member_a.sac_agent import SACAgent, SACConfig
from member_a.environment import BrainBlockEnv
from common.visualize import render_board, render_episode_replay


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC on BrainBlock")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to SAC model checkpoint")
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "cnn_mlp"])
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["sparse", "shaped"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--render-solutions", type=int, default=5,
                        help="Number of distinct solutions to render")
    parser.add_argument("--render-trace", action="store_true",
                        help="Render step-by-step trace for one episode")
    parser.add_argument("--no-masking", action="store_true",
                        help="Disable action masking (failure-mode evaluation)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from policy instead of argmax")
    return parser.parse_args()


def board_to_tiling_key(placed: list) -> frozenset:
    return frozenset(
        (piece_type, frozenset(map(tuple, cells)) if isinstance(cells, list)
         else frozenset(cells))
        for piece_type, cells in placed
    )


@torch.no_grad()
def evaluate(args):
    device = torch.device("cpu")

    config = SACConfig(
        encoder_type=args.encoder,
        hidden_dim=args.hidden_dim,
    )
    agent = SACAgent(config, device)
    agent.load(args.model)
    agent.actor.eval()

    env = BrainBlockEnv(reward_mode=args.reward)

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(args.model).parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rewards   = []
    all_lengths   = []
    all_successes = []
    all_coverages = []
    all_invalids  = []

    unique_solutions: dict = {}
    trace_episode = None

    deterministic = not args.stochastic
    mode_str = "stochastic" if args.stochastic else "deterministic"
    print(f"Evaluating {args.episodes} episodes ({mode_str})...")
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder}, Reward: {args.reward}")
    print("=" * 60)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        action_mask = info["action_mask"]

        ep_reward = 0.0
        ep_length = 0
        done = False

        board_snapshots = []
        piece_history   = []

        while not done:
            grid = torch.tensor(obs["grid"], device=device).unsqueeze(0)
            vec  = torch.tensor(obs["vec"],  device=device).unsqueeze(0)
            eff_mask = (np.ones(320, dtype=np.float32) if args.no_masking
                        else action_mask.astype(np.float32))
            mask = torch.tensor(eff_mask, device=device).unsqueeze(0)

            probs, _ = agent.actor(grid, vec, mask)
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                action = torch.multinomial(probs, num_samples=1).squeeze().item()

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            board_snapshots.append(env.board.copy())
            if env._placed:
                piece_history.append(env._placed[-1][0])

            if done:
                reason = next_info.get("termination_reason", "unknown")
                all_rewards.append(ep_reward)
                all_lengths.append(ep_length)
                all_successes.append(1.0 if reason == "success" else 0.0)
                all_coverages.append(next_info.get("coverage", 0.0))
                all_invalids.append(1.0 if reason == "illegal_action" else 0.0)

                if reason == "success":
                    key = board_to_tiling_key(env._placed)
                    if key not in unique_solutions:
                        unique_solutions[key] = [
                            (pt, set(cells)) for pt, cells in env._placed
                        ]
                        print(f"  Episode {ep}: Found solution #{len(unique_solutions)}")

                if reason == "success" and trace_episode is None:
                    trace_episode = (board_snapshots, piece_history)
            else:
                obs         = next_obs
                action_mask = next_info["action_mask"]

    # ── Summary ────────────────────────────────────────────────────
    print("=" * 60)
    success_rate  = np.mean(all_successes)
    mean_reward   = np.mean(all_rewards)
    std_reward    = np.std(all_rewards)
    mean_length   = np.mean(all_lengths)
    invalid_rate  = np.mean(all_invalids)
    mean_coverage = np.mean(all_coverages)

    results = {
        "episodes":              args.episodes,
        "mode":                  mode_str,
        "success_rate":          float(success_rate),
        "mean_reward":           float(mean_reward),
        "std_reward":            float(std_reward),
        "mean_length":           float(mean_length),
        "invalid_rate":          float(invalid_rate),
        "mean_coverage":         float(mean_coverage),
        "unique_solutions_found": len(unique_solutions),
    }

    print(f"Success rate:     {success_rate:.4f}")
    print(f"Mean reward:      {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"Mean ep length:   {mean_length:.2f}")
    print(f"Invalid-act rate: {invalid_rate:.4f}")
    print(f"Mean coverage:    {mean_coverage:.4f}")
    print(f"Unique solutions: {len(unique_solutions)}")

    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    solutions_json = [
        {"solution_id": i + 1,
         "placed": [{"piece": pt, "cells": sorted([list(c) for c in cells])}
                    for pt, cells in placed]}
        for i, placed in enumerate(unique_solutions.values())
    ]
    with open(out_dir / "eval_solutions.json", "w") as f:
        json.dump(solutions_json, f, indent=2)

    # ── Render distinct solutions ──────────────────────────────────
    n_render      = min(args.render_solutions, len(unique_solutions))
    solution_list = list(unique_solutions.values())
    for i in range(n_render):
        save_path = str(out_dir / f"solution_{i+1}.png")
        render_board(solution_list[i], title=f"Solution #{i+1}",
                     show=False, save_path=save_path)
        print(f"  Saved: {save_path}")

    # ── Render step-by-step trace ──────────────────────────────────
    if args.render_trace and trace_episode is not None:
        snaps, pieces = trace_episode
        save_path = str(out_dir / "episode_trace.png")
        render_episode_replay(snaps, pieces, save_path=save_path, show=False)
        print(f"  Saved trace: {save_path}")

    print(f"\nResults saved to: {out_dir}")
    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
