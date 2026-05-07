# Chain Reaction Design

Five months in the freezer. It is May. The goal now is not to admire the architecture; it is to thaw it, verify it, and ship a playable nostalgic prototype.

## Purpose

This repository implements a high-performance Chain Reaction environment for three consumers:

1. a TypeScript rules prototype for human-readable verification;
2. a zero-dependency C++ core for deterministic execution;
3. thin Python/PufferLib and Godot bindings that consume the same core.

The architecture exists to keep gameplay semantics in exactly one place while allowing massive reinforcement-learning throughput and a lightweight renderer.

## Stack

- **Logic Definition:** TypeScript. Strictly typed, easy to debug, and useful as high-quality prompt context for AI translation.
- **Engine:** C++. A single header, STB-style, zero dependencies, no heap allocation in the core game loop.
- **Dojo:** Python / PyTorch PPO using PufferLib Ocean as the vectorized environment engine.
- **Glass:** Godot 4.x through GDExtension.

## Shared Core Philosophy

The C++ header is the source of runtime truth. Godot and PufferLib are consumers, not owners, of game logic.

- Godot must remain a dumb terminal: read state, render sprites, pass user actions, display outcomes.
- PufferLib must remain a throughput harness: create many environments, step them, collect observations and rewards. The learning code may be repo-owned PyTorch so the model, masking, and PPO semantics stay inspectable.
- Neither consumer may reimplement explosion, ownership, legality, terminal-state, reward, or winner logic.

If behavior differs between consumers, the C++ header is wrong, the binding is wrong, or the test is wrong. There is no fourth option where duplicated logic gets to live because it is convenient.

## Build Pipeline

This is not a monolith. It is a pipeline.

1. The TypeScript prototype verifies rules and documents intent. It is not production code.
2. The C++ header is generated or hand-aligned from the verified TypeScript logic.
3. The same C++ header is compiled by `setup.py` / Cython for Python and PufferLib.
4. The same C++ header is compiled by `SCons` for the Godot GDExtension.

The header is compiled twice, but the rules exist once.

## State Model

Compute over memory. State is cheap. Allocation is fatal.

The board is flattened into fixed-size 1D arrays indexed as:

```text
index = y * WIDTH + x
```

The core uses fixed-width integer arrays suitable for cache-local stepping and environment duplication:

- token counts fit in `int8_t` / `uint8_t`;
- owners fit in `int8_t` / `uint8_t`;
- critical mass is computed from coordinates or exposed as a fixed map;
- wave logs are fixed-size flat arrays in a separate optional `WaveLog`, never heap-backed containers;
- no `std::vector`, `std::map`, heap allocation, exceptions, RTTI, or dependency-bearing STL structures in the core loop.

PufferLib must be able to duplicate thousands of environments per CPU core without fighting allocator churn or pointer-heavy memory layouts.

## Gameplay Rules

This is a deterministic, perfect-information variant of classic Atoms.

- **Board:** 8x8 grid.
- **Cell state:** token count plus owner ID: `0=None`, `1=P1`, `2=P2`.
- **Critical mass:** corners explode at 2, edges at 3, center cells at 4.
- **Legal move:** a player may increment an empty cell or a cell they already own.
- **Illegal move:** rejected deterministically; it must not partially mutate state.
- **Explosion:** a critical cell sends one unit of pressure to each orthogonal neighbor.
- **Player turns:** players act one at a time. The acting player chooses one legal field, the full cascade resolves to a stable board, then the next player acts. Simultaneity is a cascade-wave property, not simultaneous player action selection.
- **Turn boundary:** a single turn ends when all cascade waves have stabilized or when a wave leaves exactly one player with tokens. Elimination is terminal; the core does not continue resolving owner-only fireworks after the opponent is gone.
- **Alive state:** all participating players start alive. After a move that caused at least one explosion, the core recomputes the alive-player mask from board ownership. Players with no remaining tokens are eliminated and may not move.
- **Win condition:** winner is interpretation, not stored simulation state. If the alive-player mask has exactly one bit set, that player has won. A placement without explosion does not recompute eliminations, so the opening transient is safe without a separate seen-player mask.
- **No draw/max-turn rule:** core gameplay allows unbounded play. Training harnesses may impose high episode caps for batch hygiene, but that is truncation, not a game-rule draw.

## Explosion Semantics

Explosions are simultaneous, not sequential.

When a move causes a cascade, all currently critical cells explode into a next buffer before the next wave is evaluated. This double-buffering rule is mandatory because sequential in-place propagation creates directional bias: the result can depend on scan order rather than board state.

The required cascade loop shape is:

1. copy or derive a clean next buffer from the current wave;
2. mark all cells that are critical at the start of the wave;
3. remove their critical mass simultaneously;
4. collect outgoing pressure by recipient and owner;
5. apply pressure simultaneously: same-owner pressure stacks, opposing two-player pressure cancels, and any nonzero net pressure adds tokens and captures the target for the net owner;
6. swap buffers;
7. if exactly one player owns tokens after the wave, stop and declare that player the winner;
8. otherwise repeat until no cells are critical.

Any implementation that mutates a single board while scanning it must be treated as suspicious until proven equivalent. It probably is not. That little goblin has teeth.

Cells made critical by incoming pressure do not explode until the next wave. This keeps each wave graph-like: inputs are measured from one stable board, pressure is accumulated independently, and outputs become the next board.

Opposing simultaneous pressure must not resolve by scan order. For the two-player MVP, equal opposing pressure cancels at the target. Unequal opposing pressure leaves only the net pressure, owned by the stronger side. This keeps engine iteration order from deciding ownership.

In normal alternating legal play, stable boards should not contain opponent-owned critical cells waiting to explode. The opposing-pressure fixtures still matter: they verify the wave resolver as a mathematical transform over arbitrary wave states and protect symmetry under board rotations/flips instead of only testing common reachable trajectories.

For the MVP, cascades resolve until stable or terminal elimination without a maximum wave guard. Infinite or very long cascades before elimination are accepted as part of the terrain rather than hidden behind an arbitrary cutoff. If this becomes a runtime problem, the fix must be explicit: expose a bounded stepping mode or error state, not silently alter physics.

The core can record a fixed-size cascade flight recorder for the most recent move through `cr_step_with_log`: per-wave exploding source cells plus the post-wave token and owner arrays. The plain `cr_step` path stays lean for training throughput. If a cascade exceeds `CR_MAX_LOGGED_WAVES`, simulation still resolves to stability; only the log truncates and sets `wave_log_truncated`. A visualization cap must never become a physics cap by accident.

## Simulation vs Observation Schema

The simulation schema is the source of truth: separate flat arrays for token counts and owner IDs. It can represent transient overfull cells during cascade resolution, such as a center with five tokens before the next wave consumes its critical mass.

The observation schema is derived from stable simulation states for training: signed distance to explosion from the acting player's perspective. Empty cells are `0`. A current-player-owned cell is positive: `critical_mass - tokens`, so `1` means one token from explosion and immediate tactical opportunity. An opponent-owned cell is the negative of that same distance, so `-1` means one token from explosion and immediate danger. It is not the storage model and must not replace the simulation arrays. Consumers may read observations, but they must not infer or mutate game rules from them.

## Product Decisions

We are optimizing for a one-day development cycle to reach a playable prototype.

- **Visuals:** Godot renders the state array. No shader cleverness until the AI can beat a human.
- **Training:** the primary model-iteration path is repo-owned PyTorch PPO with the exact tiny CNN described below and, later, history-pool self-play. Native PufferLib remains a fast environment/native-trainer experiment, not the source of truth for architecture iteration. Do not claim native CNN parity unless the runtime model print and code path match this document.
- **Observation:** one spatial channel is enough for the first cut: signed distance to explosion, `(critical_mass - current_tokens) * owner_sign`, with `owner_sign` measured from the acting player's perspective. The network gets the physics directly instead of wasting capacity memorizing corner and edge geometry.
- **Fun Factor:** Godot inference uses temperature scaling. Same weights, higher entropy. The goal is adjustable personality: from Terminator to distracted sibling.

## Reinforcement Learning Strategy

Training starts from scratch. No human data, supervised fine-tuning, GANs, or theatrical model stew.

- **Algorithm:** PPO through self-play. Use PufferLib Ocean for vectorized stepping. The repo-owned Torch path is the readable reference and active development target for masking, negamax GAE, checkpoints, metrics, and the CNN architecture. Native CUDA is demoted to an optimization/research branch until the Torch loop learns with the intended model.
- **League:** intended: one current policy trains against a history pool of older checkpoints. Current native reality: single-policy self-play only; no checkpoint-history opponent sampling and no frozen opponent weights during rollout.
- **Action masking:** illegal moves are masked before softmax by setting their logits or log probabilities to negative infinity. Compute goes toward strategy, not relearning legality.
- **Value semantics:** observations are always from the player-to-move perspective, so `V(s)` means value for the current player. Because turns alternate, the next nonterminal value is the opponent's value from the current player's perspective and must be negated in GAE/bootstrapping. Standard single-agent PPO bootstrapping without this negamax sign flip teaches the critic that the opponent's good future is also good for us.
- **Self-play reward convention:** completed games are logged as winner (+1) and loser (-1) entries for dashboard accounting. Rollout reward still belongs to the actor of the terminal transition; the previous losing move depends on the alternating-turn negamax return path. This is plausible but should not be called proven until a focused trajectory fixture validates terminal credit assignment from the losing player's perspective.
- **History:** none. The game is Markovian; the current board contains the required state.
- **Episode cap:** training may use a high maximum step count to keep batches finite. Hitting that cap is a harness truncation signal, not a core draw condition.
- **PufferLib v4 integration:** environment stepping uses PufferLib's Ocean native environment contract, not the legacy Python `PufferEnv` wrapper. This repository vendors PufferLib v4 as a git submodule (`vendor/PufferLib`, branch `chain-reaction-native`) with hardcoded CUDA patches for native legal masking and negamax GAE experiments — no runtime source mutation, no boolean feature flags. The Compose setup points the PufferTank container at the submodule as its PUFFER_ROOT, so the container builds from the fork. This is infrastructure, not the model-iteration surface: Chain Reaction architecture work belongs in `training/torch_ppo/` unless we first add a clean root-owned native model extension seam.
- **Development target:** first meaningful training belongs on the CUDA workstation or PufferTank Docker, not the M1 laptop. The laptop is for rule verification, Cython smoke tests, and occasional CPU build probes. Do not tune the neural net around macOS CPU constraints.
- **Native trainer verification:** the host is not the source of truth for CUDA viability. Submodule changes must be injected through the PufferTank image by bind-mounting this repo, rebuilding inside the image, and running at least a tiny finite smoke. A build/smoke pass is necessary but not sufficient: native semantic patches still need code review or focused kernel fixtures, especially where PufferLib selects different scalar/vector paths based on horizon alignment.
- **Container artifact boundary:** Compose plus bind mounts is the development contract because it guarantees the image builds the current working tree and submodule checkout. A derived Dockerfile from `pufferai/puffertank:4.0` is appropriate for CI or release freezing, but it must not replace the edit/review loop until native semantics are stable; copied source layers are too easy to make stale during CUDA patch review.
- **Next product loop:** prove one minimal end-to-end training run before polishing presentation. After the first checkpoint can be trained, saved, and loaded, build a cheap CLI/TUI replay/debug viewer before Godot so policy behavior, legal masks, rewards, terminal states, and cascade depths are inspectable without renderer ceremony.

## Training Pivot Back To Torch

Native PufferLib model work is paused as the primary route. The attempt to reproduce the design CNN inside PufferLib exposed the wrong coupling: custom architectures currently require editing the PufferLib fork, rebuilding CUDA internals, and debugging model math, precision, PPO semantics, and kernel plumbing at the same time. That loop is too slow and too opaque for the current milestone.

The next learning milestone is therefore the repo-owned Torch PPO trainer:

1. Treat `training/torch_ppo/model.py` as the canonical network implementation.
2. Run through PufferTank so environment stepping still uses the same Ocean/Chain Reaction backend.
3. First prove short finite runs with zero illegal sampled actions, finite losses, and JSON/W&B metrics.
4. Then run checkpoint progression at a terminal-reaching cap (`max_turns >= 128`) and evaluate each checkpoint against random legal play under the same cap.
5. Only after the exact design CNN learns measurably should we revisit native acceleration, Triton fusion, or a PufferLib extension seam.

The native Puffer CNN checkpoint is a reproducible experiment, not the product direction. Its current shape is smaller than the design model and feeds PufferLib's MinGRU/default decoder rather than using spatial policy/value heads. It should not drive architecture decisions.

## Neural Network Shape

The first model should be microscopic and spatially honest. The point is not to optimize a generic baseline; it is to make the model match the board and remain readable enough to learn from.

Audit correction: this section describes the intended repo Torch CNN, implemented in `training/torch_ppo/model.py`. The native PufferLib path must not be treated as equivalent unless it implements the same 32-channel, four-block, GroupNorm residual trunk with spatial policy/value heads. Runtime logs print the actual native model; trust the logs over stale intent.

The first repo-owned training run should use this baseline unchanged unless it cannot execute. Its job is to prove the full data path, not to be strong. Tune architecture only after rollout collection, legal masking, sign-aware value targets, rewards, checkpointing, and policy inspection are visibly working.

- **Input:** `(Batch, 1, 8, 8)` signed-distance-to-explosion channel.
- **Stem:** `Conv2d(1, 32, kernel=3, padding=1)`.
- **Trunk:** four pre-activation residual blocks at constant 32 channels: `GroupNorm(8, 32)` -> `SiLU` -> `Conv2d(32, 32, 3, padding=1)` -> `GroupNorm(8, 32)` -> `SiLU` -> `Conv2d(32, 32, 3, padding=1)` -> add to the stream. No stride, pooling, or dimensionality reduction in the trunk; the board stays an 8x8 board.
- **Policy head:** `GroupNorm` -> `SiLU` -> `Conv2d(32, 1, kernel=1)` -> flatten the final 8x8 map into 64 action logits. Legal masking happens on logits before the categorical distribution is built, never by zeroing probabilities after softmax.
- **Critic head:** `GroupNorm` -> `SiLU` -> `Conv2d(32, C_v, kernel=1)` with small `C_v` such as 4 or 8 -> global average pool -> one small MLP -> scalar value. Do not apply `tanh` in the baseline; terminal rewards are bounded, but value targets and bootstrapping should not be saturated by architecture.

The policy avoids flattening the trunk into an MLP because actions are cells. A per-cell `1x1` projection keeps the action semantics aligned with board locations. The value head may aggregate globally because it predicts the whole position, not a move at one square.

Square 3x3 convolutions mix diagonal cells even though Chain Reaction pressure is orthogonal. That is acceptable for the first baseline because the kernel is standard, cheap, and can learn to ignore diagonal correlations. A cross-shaped convolution or attention-based cell mixer is an ablation after the baseline works, not the starting tax.

We reject 1D convolutions because they destroy spatial adjacency. We defer Transformers and attention residual mixers for the first cut because a tiny CNN should already cover the 8x8 board and teach the full RL loop with less ceremony.

## Godot UX

An optimal RL agent is usually a miserable houseguest. Difficulty is adjusted at inference, not by damaging training.

Godot samples moves from the policy with a temperature parameter applied to softmax:

- **Hard:** `T -> 0`; choose the strongest move.
- **Medium:** `T = 1.0`; natural policy sampling.
- **Easy:** `T >= 2.0`; noisy, exploitable, sibling-brained play.

## Decision Rule

Every technical decision must answer one of these questions:

- Does it preserve the shared C++ core as the single behavioral source of truth?
- Does it improve deterministic verification of the rules?
- Does it improve PufferLib throughput without damaging semantics?
- Does it keep Godot as a renderer instead of a second game engine?
- Does it move the one-day playable prototype closer?

If the answer is no, refactor it out or leave it unborn.
