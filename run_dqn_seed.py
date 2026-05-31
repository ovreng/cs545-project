import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    args = parser.parse_args()

    seed = args.seed
    total_steps = args.total_timesteps
    encoder = "mlp"

    for config in [("sparse", "r1"), ("shaped", "r2")]:
        reward = config[0]
        reward_short = config[1]
        out_dir = f"results/dqn_{reward_short}_{encoder}_seed{seed}"
        
        # Train
        train_cmd = [
            sys.executable, "-m", "member_b2.train",
            "--reward", reward,
            "--encoder", encoder,
            "--seed", str(seed),
            "--total-timesteps", str(total_steps),
            "--output-dir", out_dir,
            "--log-freq", "10000"
        ]
        run_cmd(train_cmd, f"{reward_short.upper()} Training for Seed {seed}")
        
        # Eval
        eval_cmd = [
            sys.executable, "-m", "member_b2.evaluate",
            "--model", f"{out_dir}/{'best_model.pt' if reward_short == 'r2' else 'final_model.pt'}",
            "--reward", reward,
            "--encoder", encoder,
            "--episodes", "1000",
            "--seed", "999",
            "--render-solutions", "5"
        ]
        run_cmd(eval_cmd, f"{reward_short.upper()} Evaluation for Seed {seed}")

        # Plot
        plot_cmd = [
            sys.executable, "-m", "member_b2.plot_curves",
            "--results-dir", out_dir
        ]
        run_cmd(plot_cmd, f"{reward_short.upper()} Plotting for Seed {seed}")

if __name__ == "__main__":
    main()
