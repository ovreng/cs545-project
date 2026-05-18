"""
BrainBlock Gymnasium Environment — Member B pipeline.

Uses shared piece definitions from common/pieces.py.
Strictly implements Reward Function 1 (Sparse Completion Reward):
  - +1 per successful piece placement
  - +100 bonus when board fully solved
  - 0 at dead-end

Action masking is built in: info["action_mask"] is always provided.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common.pieces import (
    PIECE_TYPES,
    PIECE_IDX,
    INVENTORY,
    ORIENT_TABLE,
    VALID_ORIENTS,
    PIECE_COLORS,
    encode_action,
    decode_action,
    is_legal,
    compute_action_mask,
)


def _build_obs(board: np.ndarray, current: str, queue_tail: list[str]) -> dict:
    """
    Build observation dict.
      grid: (1, 5, 8) float32 — binary occupancy
      vec:  (10,) float32 — [one-hot current piece (5), remaining counts / 2 (5)]
    """
    grid = board[np.newaxis, :, :].astype(np.float32)

    onehot = np.zeros(5, dtype=np.float32)
    onehot[PIECE_IDX[current]] = 1.0

    counts = np.zeros(5, dtype=np.float32)
    for p in queue_tail:
        counts[PIECE_IDX[p]] += 1.0
    # Normalize counts (max 2 per piece)
    counts /= 2.0

    vec = np.concatenate([onehot, counts])
    return {"grid": grid, "vec": vec}


class BrainBlockEnv(gym.Env):
    """
    BrainBlock 8×5 tetromino packing puzzle.
    Reward Mode: Strict Sparse Completion (Member B methodology override).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, shape=(1, 5, 8), dtype=np.float32),
            "vec": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(320)

        # State (set in reset)
        self.board: np.ndarray = None
        self._queue: list[str] = None
        self._step_count: int = 0
        self._placed: list[tuple[str, set]] = []

    # ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((5, 8), dtype=np.int8)
        queue = INVENTORY.copy()
        self.np_random.shuffle(queue)
        self._queue = queue
        self._step_count = 0
        self._placed = []

        obs = _build_obs(self.board, self._current, self._tail)
        mask = compute_action_mask(self.board, self._current)
        info = {"action_mask": mask}
        return obs, info

    # ──────────────────────────────────────────────────────────────────
    @property
    def _current(self) -> str:
        return self._queue[0] if self._queue else "I"

    @property
    def _tail(self) -> list[str]:
        return self._queue[1:] if self._queue else []

    # ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        orient, x, y = decode_action(action)
        current = self._current

        # --- Legality check ---
        if not is_legal(self.board, current, orient, x, y):
            obs = _build_obs(self.board, current, self._tail)
            mask = np.zeros(320, dtype=np.int8)
            info = {"action_mask": mask, "termination_reason": "illegal_action"}
            # With action masking, we shouldn't reach here normally. 
            # If we do, penalty could be applied, but project plan says 
            # masking means 100% valid. Return 0 to avoid messing up.
            return obs, 0.0, True, False, info

        # --- Apply placement ---
        cells_placed = set()
        for dx, dy in ORIENT_TABLE[current][orient]:
            nx, ny = x + dx, y + dy
            self.board[ny, nx] = 1
            cells_placed.add((nx, ny))
        self._placed.append((current, cells_placed))

        # Advance queue
        self._queue.pop(0)
        self._step_count += 1

        # --- Terminal: success ---
        if len(self._queue) == 0:
            reward = 1.0 + 100.0  # +1 for placement, +100 for completion
            obs = _build_obs(self.board, "I", [])  # dummy; episode over
            mask = np.zeros(320, dtype=np.int8)
            info = {
                "action_mask": mask,
                "termination_reason": "success",
                "pieces_placed": self._step_count,
                "coverage": self.board.sum() / 40.0,
            }
            return obs, reward, True, False, info

        # --- Dead-end check ---
        new_mask = compute_action_mask(self.board, self._current)
        if new_mask.sum() == 0:
            reward = 1.0  # +1 for the valid placement just made, but episode ends
            obs = _build_obs(self.board, self._current, self._tail)
            info = {
                "action_mask": new_mask,
                "termination_reason": "dead_end",
                "pieces_placed": self._step_count,
                "coverage": self.board.sum() / 40.0,
            }
            return obs, reward, True, False, info

        # --- Normal step ---
        reward = 1.0  # +1 for the valid placement
        obs = _build_obs(self.board, self._current, self._tail)
        info = {
            "action_mask": new_mask,
            "pieces_placed": self._step_count,
            "coverage": self.board.sum() / 40.0,
        }
        return obs, reward, False, False, info

    # ──────────────────────────────────────────────────────────────────
    def render(self):
        # Import here to avoid requiring matplotlib at training time
        from common.visualize import render_board
        return render_board(
            self._placed,
            title=f"BrainBlock — {len(self._placed)}/10 | next: {self._current if self._queue else 'DONE'}",
            show=(self.render_mode == "human"),
        )

    def close(self):
        pass
