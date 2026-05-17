"""
BrainBlock Packing Environment — Faz 1
Gymnasium-compatible environment for the 8×5 tetromino packing puzzle.

Coordinate convention (locked in Faz 0 §2):
  board[y, x], x ∈ {0..7} horizontal, y ∈ {0..4} vertical.
  Action: a = orient*40 + x*5 + y  (bijective, |A|=320)
"""

from __future__ import annotations

import copy
from itertools import product
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# 1. Tetromino base definitions  (local coords, min dx=0, min dy=0)
# ---------------------------------------------------------------------------

PIECE_TYPES = ["I", "O", "L", "Z", "T"]

# Each piece: frozenset of (dx, dy) offsets — reference orientation
_BASE_OFFSETS: dict[str, list[tuple[int, int]]] = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "L": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
}

PIECE_COLORS = {
    "I": "#4FC3F7",  # light blue
    "O": "#FFD54F",  # amber
    "L": "#FF8A65",  # orange
    "Z": "#81C784",  # green
    "T": "#CE93D8",  # purple
    None: "#ECEFF1",  # empty cell
}


# ---------------------------------------------------------------------------
# 2. Orientation generation (D4 group)
# ---------------------------------------------------------------------------

def _normalize(offsets: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    """Translate so min dx=0, min dy=0; return as frozenset."""
    min_x = min(dx for dx, dy in offsets)
    min_y = min(dy for dx, dy in offsets)
    return frozenset((dx - min_x, dy - min_y) for dx, dy in offsets)


def _rotate90(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """90° CCW rotation: (dx, dy) → (-dy, dx)."""
    return [(-dy, dx) for dx, dy in offsets]


def _reflect(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Horizontal reflection: (dx, dy) → (-dx, dy)."""
    return [(-dx, dy) for dx, dy in offsets]


def _build_orientations(
    base: list[tuple[int, int]],
    include_reflection: bool = True,
) -> list[Optional[frozenset]]:
    """
    Generate up to 8 D4 transforms; deduplicate; return list of length 8.
    Slots with duplicate or excluded shapes are set to None (masked).

    include_reflection=False: reflection-only orientations (indices 4-7 in the
    generation order) are masked even if geometrically distinct. Use this for
    pieces like Z where the mirror (S) is treated as a separate, absent piece.
    """
    seen: list[frozenset] = []
    all8: list[Optional[frozenset]] = []

    current = base
    for _ in range(4):
        norm = _normalize(current)
        if norm not in seen:
            seen.append(norm)
        all8.append(norm)
        current = _rotate90(current)

    reflected = _reflect(base)
    for _ in range(4):
        norm = _normalize(reflected)
        if include_reflection and norm not in seen:
            seen.append(norm)
        all8.append(norm)
        reflected = _rotate90(reflected)

    result: list[Optional[frozenset]] = []
    assigned: set[frozenset] = set()
    for i, fs in enumerate(all8):
        is_reflection_slot = i >= 4
        if is_reflection_slot and not include_reflection:
            result.append(None)
        elif fs not in assigned:
            result.append(fs)
            assigned.add(fs)
        else:
            result.append(None)

    return result


# Per-piece reflection policy (Faz0 §3 design decision):
#   L: reflection included — mirror gives J, counted as valid L variant.
#   Z: reflection excluded — mirror gives S, which is not in the inventory.
#   All others: symmetric under reflection, so flag has no practical effect.
_INCLUDE_REFLECTION = {"I": True, "O": True, "L": False, "Z": False, "T": True}

# Pre-compute orientation tables at import time
# ORIENT_TABLE[piece_type][orient_idx] = frozenset of (dx,dy) or None
ORIENT_TABLE: dict[str, list[Optional[frozenset]]] = {
    p: _build_orientations(_BASE_OFFSETS[p], _INCLUDE_REFLECTION[p])
    for p in PIECE_TYPES
}

# Sorted list of valid orient indices per piece (for mask computation)
VALID_ORIENTS: dict[str, list[int]] = {
    p: [i for i, v in enumerate(ORIENT_TABLE[p]) if v is not None]
    for p in PIECE_TYPES
}


# ---------------------------------------------------------------------------
# 3. Action encode / decode  (bijective, §2)
# ---------------------------------------------------------------------------

def encode_action(orient: int, x: int, y: int) -> int:
    return orient * 40 + x * 5 + y


def decode_action(a: int) -> tuple[int, int, int]:
    orient = a // 40
    r = a % 40
    x = r // 5
    y = r % 5
    return orient, x, y


# ---------------------------------------------------------------------------
# 4. Legality check
# ---------------------------------------------------------------------------

def is_legal(board: np.ndarray, piece: str, orient: int, x: int, y: int) -> bool:
    """Return True iff placing `piece` with `orient` at anchor (x,y) is legal."""
    cells = ORIENT_TABLE[piece][orient]
    if cells is None:
        return False  # redundant orientation
    for dx, dy in cells:
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= 8 or ny < 0 or ny >= 5:
            return False
        if board[ny, nx] != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# 5. Action mask
# ---------------------------------------------------------------------------

def compute_action_mask(board: np.ndarray, piece: str) -> np.ndarray:
    """Return bool array of shape (320,) with True for every legal action."""
    mask = np.zeros(320, dtype=np.int8)
    for orient in VALID_ORIENTS[piece]:
        cells = ORIENT_TABLE[piece][orient]
        for x, y in product(range(8), range(5)):
            if all(
                0 <= x + dx < 8 and 0 <= y + dy < 5 and board[y + dy, x + dx] == 0
                for dx, dy in cells
            ):
                mask[encode_action(orient, x, y)] = 1
    return mask


# ---------------------------------------------------------------------------
# 6. Observation builder
# ---------------------------------------------------------------------------

PIECE_IDX = {p: i for i, p in enumerate(PIECE_TYPES)}


def _build_obs(board: np.ndarray, current: str, queue_tail: list[str]) -> dict:
    """
    grid : (1, 5, 8) int8
    vec  : (10,) float32
      [0:5]  current_piece_onehot
      [5:10] remaining_counts (current excluded), each /2 → [0,1]
    """
    grid = board[np.newaxis, :, :].copy()

    onehot = np.zeros(5, dtype=np.float32)
    onehot[PIECE_IDX[current]] = 1.0

    counts = np.zeros(5, dtype=np.float32)
    for p in queue_tail:
        counts[PIECE_IDX[p]] += 1.0
    counts /= 2.0  # normalize: max count per type is 2

    vec = np.concatenate([onehot, counts])
    return {"grid": grid, "vec": vec}


# ---------------------------------------------------------------------------
# 7. Gymnasium Environment
# ---------------------------------------------------------------------------

INVENTORY = ["I", "I", "O", "O", "L", "L", "Z", "Z", "T", "T"]


class BrainBlockEnv(gym.Env):
    """
    BrainBlock 8×5 tetromino packing puzzle.

    reward_mode: "sparse" → R1, "shaped" → R2 (potential-based, default).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, reward_mode: str = "shaped", render_mode: Optional[str] = None):
        super().__init__()
        assert reward_mode in ("sparse", "shaped"), f"Unknown reward_mode: {reward_mode}"
        self.reward_mode = reward_mode
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0, 1, shape=(1, 5, 8), dtype=np.int8),
            "vec": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(320)

        # state (set in reset)
        self.board: np.ndarray = None
        self._queue: list[str] = None
        self._step_count: int = 0
        # track placed pieces for render: list of (piece_type, set of (x,y))
        self._placed: list[tuple[str, set]] = []

        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------
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

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    # ------------------------------------------------------------------
    @property
    def _current(self) -> str:
        return self._queue[0]

    @property
    def _tail(self) -> list[str]:
        return self._queue[1:]

    # ------------------------------------------------------------------
    def step(self, action: int):
        orient, x, y = decode_action(action)
        current = self._current

        # --- Legality check ---
        if not is_legal(self.board, current, orient, x, y):
            obs = _build_obs(self.board, current, self._tail)
            mask = np.zeros(320, dtype=np.int8)
            info = {"action_mask": mask, "termination_reason": "illegal_action"}
            return obs, 0.0, True, False, info  # terminate, R=0

        # --- Φ before placement (for shaped reward) ---
        phi_before = self.board.sum() / 40.0

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
            reward = self._final_reward(phi_before, success=True)
            obs = _build_obs(self.board, "I", [])  # dummy; episode over
            mask = np.zeros(320, dtype=np.int8)
            info = {"action_mask": mask, "termination_reason": "success"}
            if self.render_mode == "human":
                self._render_frame()
            return obs, reward, True, False, info

        # --- Dead-end check ---
        new_mask = compute_action_mask(self.board, self._current)
        if new_mask.sum() == 0:
            reward = self._step_reward(phi_before)
            obs = _build_obs(self.board, self._current, self._tail)
            info = {"action_mask": new_mask, "termination_reason": "dead_end"}
            if self.render_mode == "human":
                self._render_frame()
            return obs, reward, True, False, info

        # --- Normal step ---
        reward = self._step_reward(phi_before)
        obs = _build_obs(self.board, self._current, self._tail)
        info = {"action_mask": new_mask}

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, False, False, info

    # ------------------------------------------------------------------
    def _step_reward(self, phi_before: float) -> float:
        if self.reward_mode == "sparse":
            return 0.0
        phi_after = self.board.sum() / 40.0
        gamma = 0.99
        return gamma * phi_after - phi_before

    def _final_reward(self, phi_before: float, success: bool) -> float:
        base = 1.0 if success else 0.0
        if self.reward_mode == "sparse":
            return base
        phi_after = self.board.sum() / 40.0
        gamma = 0.99
        return base + (gamma * phi_after - phi_before)

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        color_grid = [[PIECE_COLORS[None]] * 8 for _ in range(5)]
        for piece_type, cells in self._placed:
            for cx, cy in cells:
                color_grid[cy][cx] = PIECE_COLORS[piece_type]

        if self.render_mode == "human":
            if self._fig is None:
                plt.ion()
                self._fig, self._ax = plt.subplots(figsize=(8, 5))
            else:
                self._ax.clear()
            fig, ax = self._fig, self._ax
        else:
            fig, ax = plt.subplots(figsize=(8, 5))

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

        placed_count = len(self._placed)
        title = f"BrainBlock — {placed_count}/10 pieces placed"
        if self._queue:
            title += f" | next: {self._current}"
        ax.set_title(title, fontsize=11)

        plt.tight_layout()

        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
        else:
            fig.canvas.draw()
            plt.pause(0.01)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    # ------------------------------------------------------------------
    def render_episode(self, history: list[np.ndarray], piece_history: list[str],
                       save_path: Optional[str] = None):
        """
        Replay a full episode from saved board states.
        history: list of board snapshots (np.ndarray 5×8).
        piece_history: piece placed at each step.
        """
        n = len(history)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        axes = np.array(axes).flatten()

        placed_so_far: list[tuple[str, set]] = []
        prev_board = np.zeros((5, 8), dtype=np.int8)

        for i, (board_snap, piece_type) in enumerate(zip(history, piece_history)):
            diff = np.argwhere(board_snap - prev_board)
            new_cells = {(c, r) for r, c in diff}
            placed_so_far.append((piece_type, new_cells))
            prev_board = board_snap.copy()

            color_grid = [[PIECE_COLORS[None]] * 8 for _ in range(5)]
            for pt, cells in placed_so_far:
                for cx, cy in cells:
                    color_grid[cy][cx] = PIECE_COLORS[pt]

            ax = axes[i]
            for row in range(5):
                for col in range(8):
                    rect = mpatches.FancyBboxPatch(
                        (col, 4 - row), 1, 1,
                        boxstyle="round,pad=0.05",
                        facecolor=color_grid[row][col],
                        edgecolor="white", linewidth=1.5,
                    )
                    ax.add_patch(rect)
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"Step {i+1}: {piece_type}", fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Episode Replay", fontsize=12)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Backtracking solver (used by tests and optionally by evaluation)
# ---------------------------------------------------------------------------

def find_solution(piece_queue: list[str]) -> Optional[list[tuple[int, int, int]]]:
    """
    Find one valid sequence of (orient, x, y) placements for the given piece_queue.
    Returns list of length len(piece_queue), or None if unsolvable.
    Uses DFS with pruning.
    """
    board = np.zeros((5, 8), dtype=np.int8)
    moves: list[tuple[int, int, int]] = []

    def dfs(step: int) -> bool:
        if step == len(piece_queue):
            return board.sum() == 40

        piece = piece_queue[step]
        for orient in VALID_ORIENTS[piece]:
            cells = ORIENT_TABLE[piece][orient]
            for x, y in product(range(8), range(5)):
                if all(
                    0 <= x + dx < 8 and 0 <= y + dy < 5 and board[y + dy, x + dx] == 0
                    for dx, dy in cells
                ):
                    for dx, dy in cells:
                        board[y + dy, x + dx] = 1
                    moves.append((orient, x, y))
                    if dfs(step + 1):
                        return True
                    moves.pop()
                    for dx, dy in cells:
                        board[y + dy, x + dx] = 0
        return False

    return moves if dfs(0) else None


# ---------------------------------------------------------------------------
# 9. Unit tests
# ---------------------------------------------------------------------------

def _run_tests():
    import traceback

    passed = 0
    failed = 0

    def ok(name):
        nonlocal passed
        print(f"  PASS  {name}")
        passed += 1

    def fail(name, err):
        nonlocal failed
        print(f"  FAIL  {name}: {err}")
        failed += 1

    # --- (d) Flatten/unflatten roundtrip ---
    try:
        for a in range(320):
            orient, x, y = decode_action(a)
            assert encode_action(orient, x, y) == a, f"roundtrip failed at a={a}"
        ok("(d) flatten/unflatten bijection")
    except Exception as e:
        fail("(d) flatten/unflatten bijection", e)

    # --- (c) Action mask matches brute-force legality ---
    try:
        env = BrainBlockEnv(reward_mode="shaped")
        obs, info = env.reset(seed=42)
        mask = info["action_mask"]
        piece = env._current
        board = env.board.copy()
        for a in range(320):
            orient, x, y = decode_action(a)
            expected = int(is_legal(board, piece, orient, x, y))
            assert mask[a] == expected, (
                f"mask mismatch at a={a} (orient={orient},x={x},y={y}): "
                f"mask={mask[a]}, is_legal={expected}"
            )
        ok("(c) action mask matches brute-force legality")
    except Exception as e:
        fail("(c) action mask matches brute-force legality", e)
        traceback.print_exc()

    # --- (b) Illegal placement terminates episode ---
    try:
        env = BrainBlockEnv(reward_mode="shaped")
        _, info = env.reset(seed=0)
        # Find an action that is guaranteed illegal (0 in mask)
        mask = info["action_mask"]
        illegal_actions = np.where(mask == 0)[0]
        assert len(illegal_actions) > 0, "Expected some illegal actions on empty board"
        _, _, terminated, _, info2 = env.step(int(illegal_actions[0]))
        assert terminated, "Illegal action should terminate episode"
        assert info2.get("termination_reason") == "illegal_action"
        ok("(b) illegal placement terminates")
    except Exception as e:
        fail("(b) illegal placement terminates", e)
        traceback.print_exc()

    # --- (a) Known solution plays to completion ---
    try:
        queue = ["I", "I", "O", "O", "L", "L", "Z", "Z", "T", "T"]
        solution = find_solution(queue)
        assert solution is not None, "Backtracking solver found no solution"
        assert len(solution) == 10

        env = BrainBlockEnv(reward_mode="shaped")
        env.reset(seed=0)
        env._queue = queue[:]
        env.board = np.zeros((5, 8), dtype=np.int8)
        env._placed = []
        env._step_count = 0

        total_reward = 0.0
        for orient, x, y in solution:
            action = encode_action(orient, x, y)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break

        assert terminated, "Episode did not terminate after 10 placements"
        assert info.get("termination_reason") == "success", (
            f"Expected success, got: {info.get('termination_reason')}"
        )
        assert env.board.sum() == 40, "Board not fully covered"
        ok(f"(a) known solution completes (total_reward={total_reward:.3f})")
    except Exception as e:
        fail("(a) known solution completes", e)
        traceback.print_exc()

    # --- (e) Unique orientation counts ---
    try:
        expected = {"I": 2, "O": 1, "L": 4, "Z": 2, "T": 4}
        for p, exp in expected.items():
            got = len(VALID_ORIENTS[p])
            assert got == exp, f"{p}: expected {exp} unique orients, got {got}"
        ok("(e) unique orientation counts (I:2, O:1, L:4, Z:2, T:4)")
    except Exception as e:
        fail("(e) unique orientation counts", e)

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("=== BrainBlock Env — Unit Tests ===")
    success = _run_tests()

    if success:
        print("\n=== Quick render demo ===")
        env = BrainBlockEnv(reward_mode="shaped", render_mode=None)
        obs, info = env.reset(seed=42)
        print(f"Reset OK | current piece: {env._current} | legal actions: {info['action_mask'].sum()}")
        print(f"obs['grid'] shape: {obs['grid'].shape}, obs['vec']: {obs['vec']}")

        mask = info["action_mask"]
        legal = np.where(mask)[0]
        action = int(legal[0])
        orient, x, y = decode_action(action)
        print(f"\nPlaying action {action} → orient={orient}, x={x}, y={y}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"reward={reward:.4f}, terminated={terminated}")
        print(f"Board filled cells: {env.board.sum()}/40")
