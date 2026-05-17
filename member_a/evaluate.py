"""
Evaluation script for BrainBlock PPO.

Usage:
  python -m member_a.evaluate --model results/mlp_shaped_mask_seed42/final_model.pt \
                               --encoder mlp --reward shaped --episodes 1000 --seed 42

Features:
  - Deterministic rollouts (argmax policy)
  - Collects success rate, mean return, mean episode length, invalid-action rate
  - Finds and visualizes distinct solutions
  - Generates qualitative step-by-step rollout trace
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from member_a.agent import PPOAgent, PPOConfig
from member_a.environment import BrainBlockEnv
from common.pieces import encode_action, decode_action, ORIENT_TABLE, PIECE_IDX
from common.visualize import render_board, render_episode_replay


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on BrainBlock")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--encoder", type=str, default="mlp",
                        choices=["mlp", "cnn_mlp"])
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["sparse", "shaped"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--render-solutions", type=int, default=5,
                        help="Number of distinct solutions to render")
    parser.add_argument("--render-trace", action="store_true",
                        help="Render step-by-step trace for one episode")
    return parser.parse_args()


def board_to_tiling_key(placed: list[tuple[str, set]]) -> frozenset:
    """
    Convert placed pieces to a canonical tiling representation.
    A tiling is a frozenset of (piece_type, frozenset of (x, y) cells).
    Placement order does NOT matter.
    """
    return frozenset(
        (piece_type, frozenset(cells))
        for piece_type, cells in placed
    )


@torch.no_grad()
def evaluate(args):
    device = torch.device("cpu")

    # Build agent
    config = PPOConfig(
        encoder_type=args.encoder,
        hidden_dim=args.hidden_dim,
    )
    agent = PPOAgent(config, device)
    agent.load(args.model)
    agent.network.eval()

    # Environment
    env = BrainBlockEnv(reward_mode=args.reward)

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.model).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    all_rewards = []
    all_lengths = []
    all_successes = []
    all_coverages = []
    all_invalids = []

    # Distinct solutions
    unique_solutions: dict[frozenset, list[tuple[str, set]]] = {}
    trace_episode = None  # store one episode trace

    print(f"Evaluating {args.episodes} episodes...")
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder}, Reward: {args.reward}")
    print("=" * 60)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        action_mask = info["action_mask"]

        ep_reward = 0.0
        ep_length = 0
        done = False

        # For trace
        board_snapshots = []
        piece_history = []

        while not done:
            # Deterministic: argmax
            grid = torch.tensor(obs["grid"], device=device).unsqueeze(0)
            vec = torch.tensor(obs["vec"], device=device).unsqueeze(0)
            mask = torch.tensor(action_mask.astype(np.float32), device=device).unsqueeze(0)

            dist, _ = agent.network(grid, vec, mask)
            action = dist.probs.argmax(dim=-1).item()

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            # Snapshot
            board_snapshots.append(env.board.copy())
            # Get current piece (before it was removed from queue — use placed history)
            if env._placed:
                piece_history.append(env._placed[-1][0])

            if done:
                reason = next_info.get("termination_reason", "unknown")
                all_rewards.append(ep_reward)
                all_lengths.append(ep_length)
                all_successes.append(1.0 if reason == "success" else 0.0)
                all_coverages.append(next_info.get("coverage", 0.0))
                all_invalids.append(1.0 if reason == "illegal_action" else 0.0)

                # Track distinct solutions
                if reason == "success":
                    key = board_to_tiling_key(env._placed)
                    if key not in unique_solutions:
                        unique_solutions[key] = [
                            (pt, set(cells)) for pt, cells in env._placed
                        ]
                        print(f"  Episode {ep}: Found solution #{len(unique_solutions)}")

                # Save first successful trace
                if reason == "success" and trace_episode is None:
                    trace_episode = (board_snapshots, piece_history)
            else:
                obs = next_obs
                action_mask = next_info["action_mask"]

    # ── Summary stats ──────────────────────────────────────────────
    print("=" * 60)
    success_rate = np.mean(all_successes)
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_length = np.mean(all_lengths)
    invalid_rate = np.mean(all_invalids)
    mean_coverage = np.mean(all_coverages)

    results = {
        "episodes": args.episodes,
        "success_rate": float(success_rate),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_length": float(mean_length),
        "invalid_rate": float(invalid_rate),
        "mean_coverage": float(mean_coverage),
        "unique_solutions_found": len(unique_solutions),
    }

    print(f"Success rate:     {success_rate:.4f}")
    print(f"Mean reward:      {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"Mean ep length:   {mean_length:.2f}")
    print(f"Invalid-act rate: {invalid_rate:.4f}")
    print(f"Mean coverage:    {mean_coverage:.4f}")
    print(f"Unique solutions: {len(unique_solutions)}")

    # Save results
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Render distinct solutions ──────────────────────────────────
    n_render = min(args.render_solutions, len(unique_solutions))
    solution_list = list(unique_solutions.values())
    for i in range(n_render):
        placed = solution_list[i]
        save_path = str(out_dir / f"solution_{i+1}.png")
        render_board(placed, title=f"Solution #{i+1}", show=False)
        # Need to use matplotlib directly to save
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        from common.pieces import PIECE_COLORS, PIECE_TYPES
        import matplotlib.patches as mpatches

        color_grid = [[PIECE_COLORS[None]] * 8 for _ in range(5)]
        for piece_type, cells in placed:
            for cx, cy in cells:
                color_grid[cy][cx] = PIECE_COLORS[piece_type]

        for row in range(5):
            for col in range(8):
                rect = mpatches.FancyBboxPatch(
                    (col, 4 - row), 1, 1,
                    boxstyle="round,pad=0.05",
                    facecolor=color_grid[row][col],
                    edgecolor="white", linewidth=2,
                )
                ax.add_patch(rect)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.axis("off")
        legend_patches = [
            mpatches.Patch(color=PIECE_COLORS[p], label=p) for p in PIECE_TYPES
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
                  bbox_to_anchor=(1.12, 1))
        ax.set_title(f"Solution #{i+1}", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
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
