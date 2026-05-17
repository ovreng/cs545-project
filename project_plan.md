# BrainBlock DRL Project — Design Choices & Task Plan

---

## 1. Design Decisions (with Pros & Cons)

### 1.1 Invalid Action Handling

When the agent selects an action that results in an illegal placement (out-of-bounds or overlap):

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A) Hard termination** | Episode ends immediately on any invalid action | Simple to implement; strong penalty signal; agent learns to avoid invalid actions quickly | Agent gets very few steps early in training → slow learning; harsh — one mistake ends everything |
| **B) Negative reward + skip** | Give a negative reward, skip the turn, keep the same piece | Agent keeps playing → more learning signal per episode; more forgiving during exploration | Agent may learn to "farm" negative rewards if reward isn't tuned carefully; longer episodes, slower wall-clock training |
| **C) Action masking** | Mask out invalid actions so the agent can never pick them | No wasted steps; 100% valid trajectories; fastest learning | More complex implementation; must compute a valid-action mask at every step; policy network needs masking logic |
| **D) Negative reward + retry (same piece, no skip)** | Agent retries until it picks a valid action (with a retry cap) | Agent always progresses eventually; good exploration | Can get stuck in retry loops; need a retry limit which adds a hyperparameter |

> **Recommendation:** Start with **C (action masking)** for the main experiments. It is the cleanest and most efficient. Optionally compare with **A (hard termination)** as a second reward-function experiment since the assignment requires two reward functions anyway.

---

### 1.2 State Representation

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A) Multi-channel binary grid** | Channel 0: board occupancy (8×5 binary). Channels 1–5: one-hot current piece type. Channel 6+: remaining piece counts encoded as a small grid or vector | Spatial structure preserved → works well with CNNs; clean separation of information | Small board (8×5) means very small feature maps — CNN may be overkill |
| **B) Flat vector** | Flatten board (40 binary values) + one-hot current piece (5) + remaining inventory (5 counts) = 50-dim vector | Simple; works well with MLPs; easy to debug | Loses spatial relationships between cells |
| **C) Multi-channel grid + inventory channel** | Board occupancy as a channel, current piece shape rendered on a separate channel, remaining counts broadcast as constant-value channels | Richest spatial info; CNN can learn spatial patterns | Largest observation size; slightly more complex to implement |

> **Recommendation:** Use **A (multi-channel binary grid)** — it balances simplicity with spatial awareness. Can be consumed by both CNN and MLP architectures.

---

### 1.3 Piece Orientation Encoding

The action space has 8 orientation indices (0–7) covering rotations and reflections.

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A) All 8 orientations for every piece** | Every piece has actions for all 8 orientations, even if some are duplicates due to symmetry | Uniform action space (always 320 actions); simple implementation | Redundant actions waste exploration; e.g., O-piece has only 1 unique orientation but 8 action slots |
| **B) Map redundant orientations to canonical ones** | Internally deduplicate: O→1, I→2, Z→2, L→4, T→4 unique orientations; redundant indices point to the same transform | No wasted action slots (if combined with masking); cleaner | Still 320 actions in the interface; just masks more; slightly harder to implement |

> **Recommendation:** Use **A** for the action interface (required by spec) but implement **B** internally — mask out redundant orientation indices so the agent never picks them. This satisfies the spec while being efficient.

---

### 1.4 Reward Function Designs (must have ≥ 2)

#### Reward Function 1: Sparse Completion Reward

| Component | Value |
|-----------|-------|
| Successful piece placement | +1 |
| Board fully solved (all 10 pieces placed) | +100 bonus |
| Invalid action (if not masked) | −10, episode ends |
| Dead-end (no valid placements left) | 0, episode ends |

- **Pros:** Simple; clear success signal; widely used in RL literature.
- **Cons:** Very sparse — agent may never see the +100 bonus early in training; slow convergence.

#### Reward Function 2: Dense Shaped Reward

| Component | Value |
|-----------|-------|
| Successful piece placement | +4 (= number of cells covered) |
| Bonus for filling a complete row | +5 per completed row |
| Bonus for contiguous empty region | +2 if all empty cells remain in one connected component |
| Board fully solved | +50 bonus |
| Invalid action (if not masked) | −5, episode ends |
| Dead-end | −(number of remaining empty cells) |

- **Pros:** Rich gradient signal at every step; agent can learn incrementally; rewards good board states.
- **Cons:** Reward shaping may bias the agent toward suboptimal strategies (e.g., completing rows instead of overall packing); harder to tune.

#### Optional Reward Function 3: Coverage-Ratio Reward

| Component | Value |
|-----------|-------|
| Each step | +(cells_filled / 40) — i.e., reward = current coverage fraction |
| Completion | +10 bonus |

- **Pros:** Smooth, monotonically increasing signal; simple.
- **Cons:** Small differences between steps; may need scaling.

> **Recommendation:** Train and compare **Reward 1 (sparse)** vs. **Reward 2 (dense shaped)**. These are sufficiently different to satisfy the assignment requirement.

---

### 1.5 DRL Algorithm Choice

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A) DQN (Deep Q-Network)** | Value-based; discrete action space is a natural fit | Well-understood; simple to implement from scratch; works well with discrete actions and action masking | 320-action output head; can be slow to converge; no natural policy stochasticity for finding multiple solutions |
| **B) PPO (Proximal Policy Optimization)** | Policy gradient with clipped surrogate objective | State-of-the-art for many tasks; stable training; naturally stochastic policy → finds diverse solutions; handles large action spaces well | More complex to implement from scratch (need actor + critic + GAE + clipping); more hyperparameters |
| **C) A2C (Advantage Actor-Critic)** | Synchronous actor-critic | Simpler than PPO; good balance of complexity and performance | Less stable than PPO; can have high variance |
| **D) REINFORCE** | Vanilla policy gradient | Simplest policy gradient to implement | High variance; very slow convergence; unlikely to solve the puzzle reliably |

> **Recommendation:** Use **PPO** as the primary algorithm — it's robust, handles action masking well, and its stochastic policy naturally discovers multiple solutions. If time allows, compare with **DQN** as a secondary experiment.

---

### 1.6 Network Architecture

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A) MLP only** | Flatten the observation → 2–3 hidden layers (256–512 units) | Simple; fast; sufficient for small state spaces | Ignores spatial structure of the board |
| **B) CNN + MLP head** | 2–3 conv layers on the 8×5 board channels → flatten → MLP for policy/value heads | Captures spatial patterns (adjacency, holes); more principled for grid-world | Board is very small (8×5), so CNNs may not gain much; slightly more complex |
| **C) CNN + MLP with separate inventory branch** | CNN on board, separate MLP on inventory vector, concatenate before final layers | Best information separation; can learn board features and inventory features independently | Most complex; potential overkill for this problem size |

> **Recommendation:** Start with **B (CNN + MLP head)** — it's a good default for grid-based environments. If it underperforms, fall back to **A (MLP only)**.

---

### 1.7 Episode Termination Conditions

| Condition | When | Notes |
|-----------|------|-------|
| **Success** | All 10 pieces placed, board fully covered (40/40 cells) | Return large positive reward |
| **Invalid action** | Agent picks an out-of-bounds or overlapping placement | Only relevant if NOT using action masking |
| **Dead-end** | No valid actions exist for the current piece | Episode must end; return negative shaped reward based on coverage |
| **Max steps** | Step count exceeds a limit (e.g., 50 steps) | Safety net; unlikely if using action masking (max 10 valid placements) |

---

### 1.8 Piece Queue: Visible vs. Hidden

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A) Full queue visible** | Agent sees the current piece AND the order of all remaining pieces | Agent can plan ahead; richer state; potentially better strategies | Much larger state space; agent may not effectively use the lookahead info |
| **B) Current piece + remaining counts only** | Agent sees which piece to place now + how many of each type remain (no order info) | Simpler state; still enough info to make decisions; matches spec requirements | Cannot plan for specific orderings; reactive rather than proactive |

> **Recommendation:** Use **B** — the spec requires "information about remaining pieces" but doesn't require order. Counts are simpler and more generalizable.

---

### 1.9 Coordinate Convention

| Option | Description |
|--------|-------------|
| **A) Row-major (row, col)** | y-axis = row (top-to-bottom), x-axis = col (left-to-right). Action = (orientation, col, row). |
| **B) Cartesian (x, y)** | x = column (left-to-right), y = row (bottom-to-top). Action = (orientation, x, y). |

> **Recommendation:** Use **A (row-major)** — it's the NumPy convention and more natural for matrix operations. The spec says "pick one and be consistent."

---

## 2. Detailed Task List — Split for 2 Members

### Ground Rules

- Each member builds their **own complete pipeline** (environment + agent + training + eval).
- This ensures both can work fully in parallel without blocking each other.
- Common interfaces and hyperparameters are agreed upon upfront (below).
- At the end, combine the best results and write the report together.

---

### Shared Agreements (Discuss Before Starting)

- [ ] Agree on state representation: multi-channel binary grid (Option A from §1.2)
- [ ] Agree on action space: 320 flat discrete actions (8 orientations × 8 × 5)
- [ ] Agree on coordinate convention: row-major
- [ ] Agree on piece shapes (exact cell offsets for each orientation of I, O, L, Z, T)
- [ ] Agree on random seeds to use: [42, 123, 456, 789, 1024]
- [ ] Set up shared Git repo with folder structure

---

### Suggested Folder Structure

```
cs545-project/
├── common/                  # Shared utilities (piece definitions, visualization)
│   ├── pieces.py            # Piece shapes and orientation definitions
│   └── visualize.py         # Board visualization utility
├── member_A/                # Member A's full pipeline
│   ├── environment.py       # Gymnasium env (reward function 1: sparse)
│   ├── agent.py             # PPO agent from scratch
│   ├── network.py           # CNN + MLP network
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── configs/             # Hyperparameter configs
├── member_B/                # Member B's full pipeline
│   ├── environment.py       # Gymnasium env (reward function 2: dense)
│   ├── agent.py             # PPO (or DQN) agent from scratch
│   ├── network.py           # Network architecture
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── configs/
├── results/                 # Combined results for report
│   ├── figures/
│   └── metrics/
├── report/                  # LaTeX report
└── presentation/            # Slides
```

---

### Member A — Tasks

#### Phase 1: Core Environment (Days 1–3)

- [ ] Implement piece definitions (I, O, L, Z, T) with all 8 orientations each
  - Define cell offsets for each orientation
  - Handle symmetry (map duplicate orientations)
- [ ] Implement `BrainBlockEnv(gymnasium.Env)`:
  - Board state management (place/check validity)
  - Piece queue with random shuffle
  - Observation: multi-channel grid (board + current piece + remaining counts)
  - Action space: `Discrete(320)` with action masking
  - **Reward function 1: Sparse completion reward** (+1 per piece, +100 for completion)
  - Episode termination: success / dead-end
- [ ] Write unit tests for the environment (valid/invalid placements, reward, termination)
- [ ] Implement board visualization (matplotlib or terminal-based)

#### Phase 2: Agent & Training (Days 4–7)

- [ ] Implement PPO agent from scratch in PyTorch:
  - Actor network (CNN + MLP → 320 logits, with action mask applied)
  - Critic network (CNN + MLP → scalar value)
  - GAE (Generalized Advantage Estimation)
  - PPO clipped surrogate loss
  - Entropy bonus for exploration
- [ ] Implement training loop:
  - Rollout buffer (collect trajectories)
  - Mini-batch updates
  - Logging: episode reward, episode length, success rate, entropy, loss curves
  - Checkpointing (save model every N episodes)
- [ ] Implement configurable hyperparameters via command-line args or config file:
  - Learning rate, gamma, lambda (GAE), clip epsilon, entropy coefficient
  - Number of epochs, batch size, rollout length

#### Phase 3: Experiments & Evaluation (Days 8–10)

- [ ] Train with **5 random seeds** [42, 123, 456, 789, 1024]
- [ ] Implement evaluation script:
  - Deterministic rollouts (argmax policy)
  - Collect: success rate, mean return (± std), mean episode length, invalid-action rate
- [ ] Generate learning curve plots:
  - Total reward vs. episode
  - Covered area vs. episode
  - Episode length over time
  - Invalid-action rate over time (if applicable)
- [ ] Collect at least 5 distinct solutions (render them visually)
- [ ] One qualitative rollout analysis (step-by-step trace with board states)

#### Phase 4: Report Sections (Days 11–12)

- [ ] Write: MDP formulation & state representation justification
- [ ] Write: Sparse reward function definition, motivation, and analysis
- [ ] Write: PPO algorithm details and hyperparameters
- [ ] Write: Results section for sparse reward experiments
- [ ] Prepare sample solution visualizations with captions

---

### Member B — Tasks

#### Phase 1: Core Environment (Days 1–3)

- [ ] Implement piece definitions (I, O, L, Z, T) with all 8 orientations each
  - Define cell offsets for each orientation (can reuse/cross-check with Member A)
  - Handle symmetry (map duplicate orientations)
- [ ] Implement `BrainBlockEnv(gymnasium.Env)`:
  - Board state management (place/check validity)
  - Piece queue with random shuffle
  - Observation: multi-channel grid (board + current piece + remaining counts)
  - Action space: `Discrete(320)` with action masking
  - **Reward function 2: Dense shaped reward** (+4 per piece, row bonus, contiguity bonus, coverage penalty at dead-end)
  - Episode termination: success / dead-end
- [ ] Write unit tests for the environment
- [ ] Implement board visualization

#### Phase 2: Agent & Training (Days 4–7)

- [ ] Implement PPO (or DQN as secondary comparison) agent from scratch in PyTorch:
  - Same architecture as Member A (CNN + MLP) for fair comparison
  - If doing DQN: replay buffer, target network, epsilon-greedy with action masking
- [ ] Implement training loop with full logging
- [ ] Implement configurable hyperparameters

#### Phase 3: Experiments & Evaluation (Days 8–10)

- [ ] Train with **5 random seeds** [42, 123, 456, 789, 1024]
- [ ] Implement evaluation script (same metrics as Member A)
- [ ] Generate learning curve plots (same set as Member A)
- [ ] Collect at least 5 distinct solutions
- [ ] One qualitative rollout analysis

#### Phase 4: Report Sections (Days 11–12)

- [ ] Write: Dense reward function definition, motivation, and analysis
- [ ] Write: Reward comparison section (combine both members' results)
- [ ] Write: Failure-case analysis and discussion
- [ ] Write: Experimental setup and evaluation protocol description
- [ ] Write: Conclusions and future improvements

---

### Joint Tasks (Days 12–14)

- [ ] Merge results from both pipelines
- [ ] Create comparative figures (sparse vs. dense reward):
  - Side-by-side learning curves
  - Success rate comparison table
  - Return distribution comparison (box plots)
- [ ] Write introduction and related background (brief)
- [ ] Finalize the report PDF
- [ ] Prepare 10-minute presentation slides
- [ ] Prepare live demo:
  - Select best-performing checkpoint
  - Build a visual demo script that renders the agent solving the puzzle step-by-step
  - Test the demo on the presentation machine
- [ ] Rehearse the presentation

---

## 3. Timeline Summary (assuming ~14 working days before June 4)

| Days | Member A | Member B | Joint |
|------|----------|----------|-------|
| 1–3 | Environment + sparse reward | Environment + dense reward | — |
| 4–7 | PPO agent + training | PPO/DQN agent + training | — |
| 8–10 | Experiments + evaluation + plots | Experiments + evaluation + plots | — |
| 11–12 | Report sections (MDP, sparse reward, algo) | Report sections (dense reward, comparison, failures) | — |
| 12–14 | — | — | Merge results, finalize report, presentation, demo |

---

## 4. Key Hyperparameters to Start With

| Parameter | Suggested Value |
|-----------|----------------|
| Learning rate | 3e-4 |
| Discount (γ) | 0.99 |
| GAE λ | 0.95 |
| PPO clip ε | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Max grad norm | 0.5 |
| Rollout steps | 2048 |
| Mini-batch size | 64 |
| PPO epochs per update | 4 |
| Total training episodes | 500,000+ |
| Network hidden dim | 256 |
| CNN channels | [32, 64] |
| CNN kernel size | 3×3 with padding |
