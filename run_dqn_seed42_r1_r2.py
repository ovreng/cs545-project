import subprocess
import sys
import time

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
    encoder = "mlp"
    
    # Run R1 (sparse)
    reward = "sparse"
    out_dir_r1 = "results/dqn_r1_mlp_seed42"
    train_cmd = [
        sys.executable, "-m", "member_b2.train",
        "--reward", reward,
        "--encoder", encoder,
        "--seed", str(seed),
        "--total-timesteps", str(total_steps),
        "--output-dir", out_dir_r1,
        "--log-freq", "10000"
    ]
    run_cmd(train_cmd, "R1 Training")
    
    eval_cmd_r1 = [
        sys.executable, "-m", "member_b2.evaluate",
        "--model", f"{out_dir_r1}/best_model.pt",
        "--reward", reward,
        "--encoder", encoder,
        "--episodes", "1000",
        "--seed", "999",
        "--render-solutions", "5"
    ]
    run_cmd(eval_cmd_r1, "R1 Evaluation")

    plot_cmd_r1 = [sys.executable, "-m", "member_b2.plot_curves", "--results-dir", out_dir_r1]
    run_cmd(plot_cmd_r1, "R1 Plotting")

    # Evaluate R2 (shaped)
    out_dir_r2 = "results/dqn_r2_mlp_seed42"
    eval_cmd_r2 = [
        sys.executable, "-m", "member_b2.evaluate",
        "--model", f"{out_dir_r2}/best_model.pt",
        "--reward", "shaped",
        "--encoder", encoder,
        "--episodes", "1000",
        "--seed", "999",
        "--render-solutions", "5"
    ]
    run_cmd(eval_cmd_r2, "R2 Evaluation (to regenerate images)")
    
    plot_cmd_r2 = [sys.executable, "-m", "member_b2.plot_curves", "--results-dir", out_dir_r2]
    run_cmd(plot_cmd_r2, "R2 Plotting")

if __name__ == "__main__":
    main()
