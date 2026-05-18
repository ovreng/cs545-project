"""
Run all 5 seeds for Member B DQN experiment.

Usage (from repo root, inside venv):
  python -m member_b.run_experiments --episodes 5000
  python -m member_b.run_experiments --episodes 5000 --seeds 42 123   # subset
"""
from __future__ import annotations

import argparse
import subprocess
import sys

SEEDS = [42, 123, 456, 789, 1024]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds or SEEDS

    print(f"=== Member B DQN — {len(seeds)} seeds × {args.episodes} episodes ===")
    for i, seed in enumerate(seeds, 1):
        cmd = [
            sys.executable, "-m", "member_b.train",
            "--seed", str(seed),
            "--episodes", str(args.episodes),
        ]
        print(f"\n[{i}/{len(seeds)}] Seed {seed}")
        if args.dry_run:
            print("  CMD:", " ".join(cmd))
            continue
        result = subprocess.run(cmd)
        status = "✅" if result.returncode == 0 else "❌"
        print(f"  {status} Done (exit {result.returncode})")

    print("\n=== All runs complete ===")


if __name__ == "__main__":
    main()
