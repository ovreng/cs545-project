"""
Evaluation script for BrainBlock DQN (Member B).
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch

from member_b.agent import DQNAgent
from member_b.environment import BrainBlockEnv
from common.visualize import render_episode_replay

def evaluate(model_path: str, seed: int = 42, num_episodes: int = 10, render: bool = False, output_dir: str = None):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    env = BrainBlockEnv()
    
    agent = DQNAgent(device=device)
    agent.load(model_path)
    agent.q_net.eval()
    
    successes = 0
    total_rewards = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_episodes):
        obs, info = env.reset(seed=seed + i)
        action_mask = info["action_mask"]
        
        ep_reward = 0.0
        done = False
        board_snapshots = []
        piece_history = []
        
        while not done:
            board_snapshots.append(env.board.copy())
            piece_history.append(env._current)
            
            action = agent.act(obs, action_mask, epsilon=0.0) # Greedy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            action_mask = info.get("action_mask")
            ep_reward += reward
        
        board_snapshots.append(env.board.copy()) # final board
        total_rewards.append(ep_reward)
        if info.get("termination_reason") == "success":
            successes += 1
            
        if render and output_dir:
            save_path = os.path.join(output_dir, f"eval_ep_{i}.png")
            render_episode_replay(board_snapshots[1:], piece_history, save_path=save_path, show=False)
            
    success_rate = successes / num_episodes
    print(f"Eval over {num_episodes} episodes: Success Rate: {success_rate:.2f}, Mean Reward: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_results_b")
    args = parser.parse_args()
    
    evaluate(args.model_path, num_episodes=args.episodes, render=args.render, output_dir=args.output_dir)
