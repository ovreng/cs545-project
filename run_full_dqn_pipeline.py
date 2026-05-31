"""
Master script to run the full DQN pipeline for Member B2 (one seed).

Steps:
1. Train for 2M timesteps (Shaped reward, MLP encoder).
2. Evaluate the best model (1000 episodes, deterministic).
3. Generate learning curves.
4. Output results to results/dqn_full_seed42/
"""

import subprocess
import sys
import time
from pathlib import Path

def run_cmd(cmd, description):
    print(f"\n>>> Starting: {description}")
    print(f">>> Command: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"!!! Error: {description} failed (code {result.returncode})")
        sys.exit(1)
    print(f">>> Completed: {description} in {elapsed:.1f}s")

def main():
    seed = 42
    total_steps = 2_000_000
    reward = "shaped"
    encoder = "mlp"
    out_dir = "results/dqn_full_seed42"

    # 1. Train
    train_cmd = [
        sys.executable, "-m", "member_b2.train",
        "--reward", reward,
        "--encoder", encoder,
        "--seed", str(seed),
        "--total-timesteps", str(total_steps),
        "--output-dir", out_dir,
        "--log-freq", "10000"  # Log slightly less frequently for speed
    ]
    run_cmd(train_cmd, "Full 2M Step Training")

    # 2. Evaluate
    eval_cmd = [
        sys.executable, "-m", "member_b2.evaluate",
        "--model", f"{out_dir}/best_model.pt",
        "--reward", reward,
        "--encoder", encoder,
        "--episodes", "1000",
        "--seed", "999"  # Use different seed for eval
    ]
    run_cmd(eval_cmd, "Deterministic Evaluation (1000 eps)")

    # 3. Plot
    plot_cmd = [
        sys.executable, "-m", "member_b2.plot_curves",
        "--results-dir", out_dir
    ]
    run_cmd(plot_cmd, "Learning Curve Generation")

    print("\n" + "="*50)
    print(f"Full pipeline completed for Seed {seed}")
    print(f"Results available in: {out_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
