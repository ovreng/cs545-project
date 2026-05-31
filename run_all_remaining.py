import subprocess
import sys
import time
import os
from concurrent.futures import ProcessPoolExecutor

def run_seed(seed):
    # Set thread limits so PyTorch doesn't overload the CPU
    env_vars = os.environ.copy()
    env_vars["OMP_NUM_THREADS"] = "1"
    env_vars["MKL_NUM_THREADS"] = "1"
    env_vars["PYTHONIOENCODING"] = "utf-8"
    
    print(f">>> [Seed {seed}] Starting complete pipeline")
    
    for config in [("sparse", "r1"), ("shaped", "r2")]:
        reward, reward_short = config
        out_dir = f"results/dqn_{reward_short}_mlp_seed{seed}"
        
        # Train
        train_cmd = [
            sys.executable, "-m", "member_b2.train",
            "--reward", reward,
            "--encoder", "mlp",
            "--seed", str(seed),
            "--total-timesteps", "2000000",
            "--output-dir", out_dir,
            "--log-freq", "50000"
        ]
        
        print(f"[{seed}] ({reward_short}) Training...")
        res = subprocess.run(train_cmd, env=env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            return f"Seed {seed} failed during {reward_short} training.\nOutput:\n{res.stdout.decode()[:1000]}"

        # Evaluate
        model_name = "best_model.pt" if reward_short == "r2" else "final_model.pt"
        eval_cmd = [
            sys.executable, "-m", "member_b2.evaluate",
            "--model", f"{out_dir}/{model_name}",
            "--reward", reward,
            "--encoder", "mlp",
            "--episodes", "1000",
            "--seed", "999",
            "--render-solutions", "5"
        ]
        print(f"[{seed}] ({reward_short}) Evaluating...")
        res = subprocess.run(eval_cmd, env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            return f"Seed {seed} failed during {reward_short} evaluation."

        # Plot
        plot_cmd = [
            sys.executable, "-m", "member_b2.plot_curves",
            "--results-dir", out_dir
        ]
        print(f"[{seed}] ({reward_short}) Plotting...")
        res = subprocess.run(plot_cmd, env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            return f"Seed {seed} failed during {reward_short} plotting."
            
    print(f">>> [Seed {seed}] COMPLETED SUCCESSFULLY")
    return f"Seed {seed} complete."

def main():
    seeds = [456, 789, 1024]
    print(f"Starting concurrent execution for seeds: {seeds}")
    
    start = time.time()
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run_seed, seeds))
        
    for r in results:
        print(r)
        
    elapsed = time.time() - start
    print(f"\nAll remaining seeds finished in {elapsed/60:.1f} minutes!")

if __name__ == "__main__":
    main()
