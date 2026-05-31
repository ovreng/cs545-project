"""
Plot learning curves for Member B DQN experiments.

Reads all results_b/dqn_sparse_seed*/episode_data.npz files and
produces the 4 figures required by the assignment:
  1. Total reward vs episode
  2. Coverage (covered area) vs episode
  3. Episode length vs episode
  4. Success rate vs episode  (invalid-action rate is N/A — action masking)

Usage:
  python -m member_b.plot_curves --output-dir results_b/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = [42, 123, 456, 789, 1024]
SMOOTHING = 100   # rolling-window width


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def load_seed(seed: int, results_root: Path):
    path = results_root / f"dqn_sparse_seed{seed}" / "episode_data.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data


def plot_metric(axes, all_data, key, ylabel, title, color="steelblue", smoothing=SMOOTHING):
    """Plot mean ± std across seeds with individual runs faded."""
    min_len = min(len(d[key]) for d in all_data)
    mat = np.stack([d[key][:min_len] for d in all_data])  # (n_seeds, n_eps)

    # Smooth each seed
    smoothed = np.stack([smooth(row, smoothing) for row in mat])
    x = np.arange(smoothed.shape[1]) + smoothing

    mean = smoothed.mean(0)
    std  = smoothed.std(0)

    for row in smoothed:
        axes.plot(x, row, alpha=0.15, color=color, linewidth=0.8)

    axes.plot(x, mean, color=color, linewidth=2.0, label="Mean (5 seeds)")
    axes.fill_between(x, mean - std, mean + std, alpha=0.25, color=color, label="±1 std")
    axes.set_xlabel("Episode")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend(fontsize=8)
    axes.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="results_b")
    parser.add_argument("--output-dir",   type=str, default="results_b/figures")
    args = parser.parse_args()

    root    = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    loaded_seeds = []
    for seed in SEEDS:
        d = load_seed(seed, root)
        if d is not None:
            all_data.append(d)
            loaded_seeds.append(seed)
        else:
            print(f"  ⚠ Seed {seed} not found, skipping.")

    if not all_data:
        print("No data found. Run member_b.run_experiments first.")
        return

    print(f"Loaded {len(all_data)} seeds: {loaded_seeds}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Member B — DQN + Sparse Reward | MLP | Action Masking\n"
        f"Seeds: {loaded_seeds} | Smoothing window: {SMOOTHING} eps",
        fontsize=12, fontweight="bold"
    )

    plot_metric(axes[0, 0], all_data, "rewards",   "Episode Return",  "Total Reward vs Episode",      color="#3B82F6")
    plot_metric(axes[0, 1], all_data, "coverages",  "Coverage Ratio",  "Board Coverage vs Episode",    color="#10B981")
    plot_metric(axes[1, 0], all_data, "lengths",    "Episode Length",   "Episode Length vs Episode",    color="#F59E0B")
    plot_metric(axes[1, 1], all_data, "successes",  "Success Rate",    "Success Rate vs Episode",      color="#8B5CF6")

    plt.tight_layout()
    save_path = out_dir / "learning_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")

    # ── Per-metric individual saves ───────────────────────────────────
    metrics = [
        ("rewards",   "Episode Return",  "Total Reward vs Episode",   "#3B82F6", "reward_curve.png"),
        ("coverages", "Coverage Ratio",  "Coverage vs Episode",       "#10B981", "coverage_curve.png"),
        ("lengths",   "Episode Length",  "Episode Length vs Episode", "#F59E0B", "length_curve.png"),
        ("successes", "Success Rate",    "Success Rate vs Episode",   "#8B5CF6", "success_curve.png"),
    ]
    for key, ylabel, title, color, fname in metrics:
        fig2, ax = plt.subplots(figsize=(8, 4))
        plot_metric(ax, all_data, key, ylabel, title, color=color)
        ax.set_title(
            f"{title}\nDQN · Sparse Reward · MLP · {len(loaded_seeds)} seeds", fontsize=11
        )
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out_dir / fname}")

    # ── Summary statistics table ──────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Metric':<30} {'Mean':>10} {'Std':>10}")
    print("=" * 65)

    min_len = min(len(d["rewards"]) for d in all_data)
    for key, label, _ in [
        ("rewards",   "Mean Reward (last 100 ep)",   None),
        ("coverages", "Mean Coverage (last 100 ep)", None),
        ("lengths",   "Mean Ep Length (last 100 ep)", None),
        ("successes", "Success Rate (last 100 ep)",   None),
    ]:
        per_seed = [np.mean(d[key][-100:]) for d in all_data]
        m, s = np.mean(per_seed), np.std(per_seed)
        print(f"  {label:<28} {m:>10.4f} {s:>10.4f}")
    print("=" * 65)

    print("\nDone. Figures in:", out_dir)


if __name__ == "__main__":
    main()
