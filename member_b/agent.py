"""
DQN Agent for BrainBlock (Member B).
"""

import random
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from member_b.network import QNetwork


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done, action_mask):
        """
        obs: dict with 'grid' and 'vec'
        next_obs: dict with 'grid' and 'vec'
        action_mask: numpy array (320,) of next available actions 
                     (useful if doing double DQN, but standard DQN uses it for max Q)
        """
        self.buffer.append((obs, action, reward, next_obs, done, action_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        
        obs_grid = np.stack([b[0]["grid"] for b in batch])
        obs_vec = np.stack([b[0]["vec"] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_obs_grid = np.stack([b[3]["grid"] for b in batch])
        next_obs_vec = np.stack([b[3]["vec"] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        action_masks = np.stack([b[5] for b in batch])

        return (
            torch.FloatTensor(obs_grid),
            torch.FloatTensor(obs_vec),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_obs_grid),
            torch.FloatTensor(next_obs_vec),
            torch.FloatTensor(dones),
            torch.FloatTensor(action_masks)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_grad_norm: float = 0.5,
        replay_size: int = 100_000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        self.q_net = QNetwork(hidden_dim).to(self.device)
        self.target_net = QNetwork(hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=replay_size)

    def act(self, obs: Dict[str, np.ndarray], action_mask: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) > 0:
                return random.choice(valid_actions)
            return 0  # Fallback if no valid action (should be handled by env dead-end logic)
        
        grid = torch.FloatTensor(obs["grid"]).unsqueeze(0).to(self.device)
        vec = torch.FloatTensor(obs["vec"]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(grid, vec).cpu().numpy()[0]
        
        # Apply mask: set invalid actions to -inf
        q_values[action_mask == 0] = -np.inf
        
        # Return argmax
        # If all are masked, argmax returns 0, but usually means dead-end
        return int(np.argmax(q_values))

    def update(self, batch_size: int = 64) -> float:
        if len(self.memory) < batch_size:
            return 0.0

        grid, vec, actions, rewards, next_grid, next_vec, dones, next_masks = self.memory.sample(batch_size)
        
        grid = grid.to(self.device)
        vec = vec.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_grid = next_grid.to(self.device)
        next_vec = next_vec.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)

        # Get current Q values
        current_q = self.q_net(grid, vec).gather(1, actions).squeeze(1)

        # Get next Q values from target network
        with torch.no_grad():
            next_q_all = self.target_net(next_grid, next_vec)
            # Mask invalid next actions
            next_q_all[next_masks == 0] = -float('inf')
            
            # Use max next Q, but replace -inf with 0 if terminal
            next_q = next_q_all.max(1)[0]
            # If all actions are masked (dead-end), max is -inf. We mask those with `dones` anyway.
            # But just to be safe from NaNs:
            next_q = torch.clamp(next_q, min=-1000.0)

        # Compute expected Q values
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Compute loss (Huber Loss)
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Soft update target network
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
