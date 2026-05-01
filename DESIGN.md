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
- **Dojo:** Python / PufferLib through Cython bindings.
- **Glass:** Godot 4.x through GDExtension.

## Shared Core Philosophy

The C++ header is the source of runtime truth. Godot and PufferLib are consumers, not owners, of game logic.

- Godot must remain a dumb terminal: read state, render sprites, pass user actions, display outcomes.
- PufferLib must remain a throughput harness: create many environments, step them, collect observations and rewards.
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
- **Turn boundary:** a single turn ends only when all cascade waves have stabilized.
- **Alive state:** all participating players start alive. After a move that caused at least one explosion, the core recomputes the alive-player mask from board ownership. Players with no remaining tokens are eliminated and may not move.
- **Win condition:** winner is interpretation, not stored simulation state. If the alive-player mask has exactly one bit set, that player has won. A placement without explosion does not recompute eliminations, so the opening transient is safe without a separate seen-player mask.

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
7. repeat until no cells are critical.

Any implementation that mutates a single board while scanning it must be treated as suspicious until proven equivalent. It probably is not. That little goblin has teeth.

Cells made critical by incoming pressure do not explode until the next wave. This keeps each wave graph-like: inputs are measured from one stable board, pressure is accumulated independently, and outputs become the next board.

Opposing simultaneous pressure must not resolve by scan order. For the two-player MVP, equal opposing pressure cancels at the target. Unequal opposing pressure leaves only the net pressure, owned by the stronger side. This keeps engine iteration order from deciding ownership.

For the MVP, cascades resolve until stable without a maximum wave guard. Infinite or very long cascades are accepted as part of the terrain rather than hidden behind an arbitrary cutoff. If this becomes a runtime problem, the fix must be explicit: expose a bounded stepping mode or error state, not silently alter physics.

The core can record a fixed-size cascade flight recorder for the most recent move through `cr_step_with_log`: per-wave exploding source cells plus the post-wave token and owner arrays. The plain `cr_step` path stays lean for training throughput. If a cascade exceeds `CR_MAX_LOGGED_WAVES`, simulation still resolves to stability; only the log truncates and sets `wave_log_truncated`. A visualization cap must never become a physics cap by accident.

## Simulation vs Observation Schema

The simulation schema is the source of truth: separate flat arrays for token counts and owner IDs. It can represent transient overfull cells during cascade resolution, such as a center with five tokens before the next wave consumes its critical mass.

The observation schema is derived from stable simulation states for training: signed distance to explosion from the acting player's perspective. It is not the storage model and must not replace the simulation arrays. Consumers may read observations, but they must not infer or mutate game rules from them.

## Product Decisions

We are optimizing for a one-day development cycle to reach a playable prototype.

- **Visuals:** Godot renders the state array. No shader cleverness until the AI can beat a human.
- **Training:** single policy, self-play, history pool. No model merging. Train one tiny CNN continuously against older snapshots of itself.
- **Observation:** one spatial channel is enough for the first cut: signed distance to explosion, `(current_tokens - max_capacity) * owner_sign`. The network gets the physics directly instead of wasting capacity memorizing corner and edge geometry.
- **Fun Factor:** Godot inference uses temperature scaling. Same weights, higher entropy. The goal is adjustable personality: from Terminator to distracted sibling.

## Reinforcement Learning Strategy

Training starts from scratch. No human data, supervised fine-tuning, GANs, or theatrical model stew.

- **Algorithm:** PPO through self-play.
- **League:** one current policy trains against a history pool of older checkpoints. We do not merge models; we sample past selves to prevent catastrophic forgetting.
- **Action masking:** illegal moves are masked before softmax by setting their logits or log probabilities to negative infinity. Compute goes toward strategy, not relearning legality.
- **History:** none. The game is Markovian; the current board contains the required state.

## Neural Network Shape

The first model should be microscopic and spatially honest.

- **Input:** `(Batch, 1, 8, 8)` signed-distance-to-explosion channel.
- **Backbone:** `Conv2d(1, 32, kernel=3)` -> `ReLU` -> `Conv2d(32, 64, kernel=3)` -> `ReLU`.
- **Mixer:** flatten into `Linear(128)` for global board synthesis.
- **Actor head:** `Linear(64)` for the 64 board actions.
- **Critic head:** `Linear(1)` for value in `[-1.0, 1.0]`.

We reject 1D convolutions because they destroy spatial adjacency. We reject Transformers for the first cut because `O(N^2)` attention on an 8x8 deterministic board is a velvet hammer for a thumbtack.

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
