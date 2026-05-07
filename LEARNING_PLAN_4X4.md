# Goal: First Learned Chain Reaction Model On 4x4

Train the repo-owned Torch PPO implementation to a first credible learned model on a 4x4 Chain Reaction board: at least 90% win rate against random legal play over 1000 evaluation games, with zero illegal selected actions, low truncation, sane loss/entropy behavior, and reproducible checkpoint loading/evaluation.

This is the primary learning path. Native PufferLib model work is paused as an architecture-iteration route. PufferLib remains the environment engine and later acceleration target, not the place to evolve the Chain Reaction network.

## Why 4x4 First

The 8x8 game has long episodes, sparse terminal feedback, and enough tactical horizon that a broken training loop can look merely weak for a long time. The current native 8x8 baseline reached only about 57-58% against random legal play under corrected caps, which is not a learned model; it is a warning that training semantics, evaluation shape, or architecture ownership were still muddy.

The 4x4 board is a learning sandbox:

- Shorter games create more terminal feedback per wall-clock minute.
- Smaller action space makes random legal play easier to dominate if PPO semantics are correct.
- The exact ResNet-v2 CNN can be tested without changing the architectural idea.
- Checkpoint progression should become visible at much lower transition counts.
- If 4x4 cannot reach 90% against random within a modest budget, assume a training semantics bug before blaming architecture.

The goal is not to solve final 8x8 play. The goal is to prove the end-to-end learning loop with the intended architecture in the root repo.

## Non-Goals For This Phase

- Do not implement native PufferLib CNN parity.
- Do not add Triton before the 90% gate.
- Do not add history-pool self-play before single-policy learning is proven.
- Do not tune visual/Godot integration.
- Do not change the game rules or duplicate them outside the shared core.
- Do not simplify the model into an MLP or flatten-before-policy architecture.
- Do not optimize throughput by accepting lower semantic visibility.

## Architecture Contract

Use the same ResNet-v2 basic baseline described in `DESIGN.md`, scaled only by board size.

```text
Input:  (B, 1, 4, 4)

Stem:
  Conv2d(1, 32, kernel_size=3, padding=1)

Trunk:
  4x PreActResidualBlock at constant 32 channels:
    GroupNorm(8, 32)
    SiLU
    Conv2d(32, 32, kernel_size=3, padding=1)
    GroupNorm(8, 32)
    SiLU
    Conv2d(32, 32, kernel_size=3, padding=1)
    residual add

Policy head:
  GroupNorm(8, 32)
  SiLU
  Conv2d(32, 1, kernel_size=1)
  flatten -> 16 logits

Value head:
  GroupNorm(8, 32)
  SiLU
  Conv2d(32, C_v, kernel_size=1), C_v initially 8 unless existing code already uses another documented value
  global average pool
  small MLP
  scalar current-player value
```

The trunk must not stride, pool, or flatten. The board remains a board until the policy emits one score per cell.

Do not shrink channels just because the board is 4x4. The purpose is to test the architecture and RL loop, not to create a tiny special-case network. If throughput is unusable, report the bottleneck and ask before changing capacity.

## Core Semantics To Preserve

- Observations are signed distance to explosion from the current player's perspective.
- Empty cells are `0`.
- Current-player cells are positive: `critical_mass - tokens`.
- Opponent cells are negative: `-(critical_mass - tokens)`.
- Legal actions are exactly cells with observation value `>= 0`.
- Illegal actions are masked on logits before categorical distribution construction.
- Do not softmax first and zero probabilities after.
- `V(s)` means value for the player to move.
- Nonterminal PPO/GAE bootstrap uses negamax value semantics: next state value is the opponent's value and must be negated.
- Terminal transitions do not bootstrap.
- Training max-turn cap is a harness truncation, not a core draw rule.

## Migration To 8x8 Later

A 4x4 model is not the final agent. It is loop validation and possible pretraining.

If the architecture stays fully convolutional until the heads, many weights can transfer to 8x8:

Reusable:

- Stem convolution weights.
- Residual trunk convolution weights.
- GroupNorm parameters.
- Policy `1x1` convolution head.
- Value `1x1` projection.
- Value MLP after global average pooling.

Do not blindly reuse:

- Optimizer state.
- PPO rollout buffers or normalization state.
- Any action-shape-specific decoder if one is introduced accidentally.
- Claims of policy strength.

The 8x8 policy still needs real retraining because board geometry, episode length, cascade horizon, and tactical depth change materially. Treat transfer as initialization, not solved behavior.

Future 8x8 migration gate:

1. Initialize 8x8 from 4x4 convolutional/trunk/head weights where shapes match.
2. Reset optimizer.
3. Train 8x8 from that initialization.
4. Compare against 8x8 from scratch under the same budget.
5. Keep transfer only if it improves learning speed or final random-baseline dominance.

## Definition Of First Learned Model

A checkpoint qualifies as the first learned 4x4 model only if all gates pass.

Primary evaluation gate:

- At least 90% combined win rate against random legal play.
- Evaluation over at least 1000 games.
- Wilson lower bound should remain comfortably above 85%.
- P1 win rate at least 85%.
- P2 win rate at least 85%.

Correctness gates:

- Illegal selected actions: `0`.
- Evaluation loads a saved checkpoint rather than using in-memory training state.
- Evaluation board size and max-turn cap match the training configuration unless explicitly marked as generalization.
- Terminal rate should be above 98%.
- Truncation rate should ideally be below 2%; if higher, do not claim the 90% result until truncation is explained.

Training health gates:

- Policy loss finite.
- Value loss finite.
- Entropy finite and declining gradually, not instantly collapsing.
- KL finite and not exploding.
- No NaNs/Infs in parameters, losses, logits, values, or advantages.
- Checkpoints are written at known intervals.
- Metrics are written as durable JSON and/or W&B, not only dashboard text.

Behavioral sanity gates:

- Average game length does not explode relative to random-vs-random.
- Sampled games show legal pressure building and captures, not only random-looking drift.
- Side-specific results do not hide a first-player exploit or second-player collapse.

## Transition Budgets

Define one impression as one environment transition.

Staged budget:

1. Smoke: `8k-32k` transitions.
   - Purpose: code path, board size, shapes, masks, finite losses, checkpoint write.
   - No learning claims.

2. Pulse check: `250k` transitions.
   - Purpose: check whether win rate moves above random.
   - Expected healthy signal: 60-70% against random legal play.
   - If still near 50%, inspect semantics before increasing budget.

3. First learning attempt: `1M` transitions.
   - Expected healthy signal: clear dominance, roughly 75-85%.
   - If below 70%, inspect reward, GAE, masks, terminal/truncation, and optimizer before changing architecture.

4. First learned model gate: `5M-10M` transitions.
   - Target: stable 90%+ against random legal play over 1000 evaluation games.
   - If this budget fails on 4x4, assume a training-loop or evaluation problem first.

These are planning thresholds, not guarantees. If early metrics contradict the assumptions, stop and classify the failure instead of blindly scaling.

## Rollout Shape

Do not guess the 4x4 max-turn cap. First measure random-vs-random terminal length distribution on 4x4. Set `CHAIN_REACTION_MAX_TURNS` around a high percentile, such as p99 or p99.5, so normal games terminate while pathological games remain bounded.

Initial PPO shape, pending actual random-vs-random cap measurement:

```text
CHAIN_REACTION_TOTAL_AGENTS=512 or 1024
CHAIN_REACTION_HORIZON=32
CHAIN_REACTION_MINIBATCH_SIZE=4096 or 8192
CHAIN_REACTION_MAX_TURNS=<measured p99/p99.5 cap>
```

If the measured 4x4 episode distribution is much shorter than expected, keep horizons short enough to produce frequent updates. If episodes are unexpectedly long, inspect core dynamics and cap semantics before training.

## Execution Phases

### Phase 0: Board Size Plumbing Contract

Determine how board size is configured today.

Expected areas:

- `core/chain_reaction.hpp` board constants.
- TypeScript prototype constants, if still used for fixture parity.
- Cython bridge shape assumptions.
- Puffer Ocean binding observation/action sizes.
- `training/torch_ppo/model.py` board/action constants.
- `training/torch_ppo/masking.py` flatten/action shape assumptions.
- `training/torch_ppo/train.py` checkpoint/eval assumptions.

Preferred implementation shape:

- Keep core behavior source-of-truth in C++.
- Make 4x4 an explicit training configuration or build target, not a silent global mutation that invalidates 8x8 docs.
- If compile-time board size is necessary, document and script the 4x4 build path so future sessions do not accidentally train 8x8 while believing it is 4x4.

Stop condition:

- If board-size configurability requires a broad semantic migration, stop and present options before cutting code.

### Phase 1: Static Audit Before Changes

Before implementation, audit current Torch PPO files against this plan and `DESIGN.md`.

Required audit table:

```text
Component | Expected | Actual | Status | Fix needed
```

Components:

- `training/torch_ppo/model.py`
- `training/torch_ppo/masking.py`
- `training/torch_ppo/gae.py`
- `training/torch_ppo/train.py`
- `training/torch_ppo/puffer_vec.py`
- Puffer Ocean binding/config used by Torch training
- checkpoint evaluation path, if it exists for Torch

Do not start fixing during the audit unless the issue is purely documentary and trivial. The point is to expose the terrain first.

### Phase 2: 4x4 Implementation

Implement only the changes needed for 4x4 Torch PPO.

Likely work:

- Add/adjust board-size support in the shared core or training build path.
- Ensure observation shape becomes `(B, 1, 4, 4)` for Torch.
- Ensure action count becomes 16.
- Ensure legal masks use the 4x4 observation rows.
- Ensure model constants derive from board size or are explicitly set to 4 for this path.
- Ensure checkpoint metadata records board size.
- Ensure evaluation refuses to load a checkpoint into the wrong board size unless explicitly forced.

Do not change:

- PPO algorithm shape beyond what 4x4 requires.
- CNN capacity.
- reward convention.
- legal mask semantics.
- value sign semantics.

### Phase 3: Narrow Tests And Smokes

Before training scale, run or add narrow checks.

Model shape smoke:

- Input `(N, 64)` still works only if intentionally supported for 8x8.
- Input `(N, 16)` and/or `(N, 1, 4, 4)` produces logits `(N, 16)`.
- Values shape is `(N,)` or normalized consistently by trainer.

Mask smoke:

- Empty board: all 16 actions legal.
- Mixed board: legal mask equals `obs >= 0`.
- Opponent-owned cells are never sampled.

GAE smoke:

- Terminal transition has no bootstrap.
- Nonterminal transition negates `V(next)`.
- Truncation behavior is explicit and tested.

Random-vs-random cap probe:

- Run enough 4x4 random legal games to estimate terminal length distribution.
- Record mean, median, p95, p99, p99.5, max, terminal rate under candidate cap.
- Set training cap from this measurement.

### Phase 4: Tiny Torch Smoke

Run through PufferTank, not ambient host Python.

Template command, adjusting max-turn cap after the probe:

```bash
CHAIN_REACTION_TOTAL_TIMESTEPS=8192 \
CHAIN_REACTION_TOTAL_AGENTS=256 \
CHAIN_REACTION_HORIZON=32 \
CHAIN_REACTION_MINIBATCH_SIZE=2048 \
CHAIN_REACTION_MAX_TURNS=<measured-cap> \
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer \
  bash /workspace/chain-reaction/training/torch_ppo/train.sh
```

Pass criteria:

- Clean exit.
- Finite losses.
- Illegal selected actions = 0.
- Checkpoint written.
- Board size and action count printed/logged.
- Terminal/truncation metrics visible, or lack of visibility recorded as the next fix.

### Phase 5: Pulse And Learning Runs

Pulse run:

```text
~250k transitions
```

Expected result:

- Some movement above random, ideally 60-70% against random legal play.

First learning attempt:

```text
~1M transitions
```

Expected result:

- Clear dominance, ideally 75-85%.

Gate run:

```text
5M-10M transitions
```

Expected result:

- 90%+ combined win rate against random legal play over 1000 games, satisfying side-specific and truncation gates.

At each stage, evaluate saved checkpoints, not live training state.

### Phase 6: Evaluation Report

For every meaningful checkpoint, produce a table with:

```text
checkpoint
train transitions
board size
max_turns
eval games
combined winrate
P1 winrate
P2 winrate
Wilson lower bound
illegal selected actions
truncations
terminal rate
mean episode length
median episode length
mean cascade depth if available
notes
```

Do not call a checkpoint learned without this table.

## Failure Classification

If a phase fails, classify before fixing.

- Shape failure: board size/action count/logit/value dimensions disagree.
- Mask failure: illegal moves sampled or mask differs from `obs >= 0`.
- Value failure: GAE sign, terminal bootstrap, or truncation handling wrong.
- Cap failure: training/eval dominated by truncations.
- Throughput failure: correct but too slow to reach budget.
- Optimization failure: loss/KL/entropy unstable despite correct semantics.
- Architecture failure: semantics verified, enough impressions, still no learning.

Default response order:

1. Fix shape/mask/value/cap failures before touching architecture.
2. Fix instrumentation gaps before making strength claims.
3. Only consider architecture changes after 4x4 semantics and budget are verified.
4. Only consider Triton/native acceleration after 90% is achieved or after a clearly measured throughput failure blocks reaching the budget.

## Triton And Throughput Policy

RL breathes through throughput: more correct impressions reduce noise. This project should not treat speed as vanity.

However, optimization must have a known-good target. For this phase:

- Do not add Triton before the 90% random-baseline gate unless Torch throughput is so low that the planned budget is unreachable.
- After the 90% gate, inspect profiler traces and consider Triton for any fusable operation, not only obvious bottlenecks.
- Good Triton candidates may include mask/logit transforms, advantage/return computation, batched tensor transforms, and possibly tiny CNN blocks if launch overhead dominates.
- Native PufferLib acceleration should be compared against the Torch reference, not developed as a parallel semantics experiment.

## Session Handoff Instructions

Next sessions should not rediscover or replan this direction from scratch. Start here:

1. Read this file.
2. Confirm current git status is clean or identify unrelated work.
3. Begin with Phase 0 board-size plumbing contract and Phase 1 static audit.
4. Report the audit table before changing code.
5. If board-size support requires a broad core/config design choice, stop and ask with options.
6. Otherwise implement the smallest 4x4 Torch PPO path that preserves the invariants above.

The captain's chosen direction is: 4x4 first, exact ResNet-v2 CNN, Torch PPO primary, 90% against random legal play before optimization or native model work.
