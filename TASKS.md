# TASKS.md

Atomic execution steps only. Each item should be binary: done or not done. No vague investigation tasks.

## Phase 1 — Finalize TypeScript Prototype

- [x] Implement `ChainReaction.reset()` in `core/game.ts`.
- [x] Implement legal move validation in `core/game.ts`.
- [x] Implement `ChainReaction.step(actionIndex, playerId)` in `core/game.ts`.
- [x] Implement double-buffered simultaneous cascade resolution in `core/game.ts`.
- [x] Implement winner detection in `core/game.ts`.
- [x] Ensure winner detection does not treat the first placed token as a win.
- [x] Record that core gameplay has no draw or max-turn handling; training may impose an episode cap.
- [x] Add TypeScript fixtures for corner critical mass.
- [x] Add TypeScript fixtures for edge critical mass.
- [x] Add TypeScript fixtures for center critical mass.
- [x] Add TypeScript fixtures for simultaneous multi-cell explosions.
- [x] Add TypeScript fixtures for ownership transfer during cascades.
- [x] Add TypeScript fixtures for opposing simultaneous pressure cancellation.
- [x] Add TypeScript fixtures for source owner clearing and residual owner preservation.
- [x] Add TypeScript fixtures for illegal moves.

## Phase 2 — Generate / Align C++ Header

- [x] Replace `core/chain_reaction.hpp` stub with complete fixed-array implementation.
- [x] Define board constants in `core/chain_reaction.hpp`.
- [x] Define fixed-size token and owner arrays in `core/chain_reaction.hpp`.
- [x] Implement critical mass calculation in `core/chain_reaction.hpp`.
- [x] Implement legal move validation in `core/chain_reaction.hpp`.
- [x] Implement double-buffered simultaneous cascade resolution in `core/chain_reaction.hpp`.
- [x] Implement winner detection in `core/chain_reaction.hpp`.
- [x] Implement signed-distance-to-explosion observation export in `core/chain_reaction.hpp`.
- [x] Implement legal-action mask export in `core/chain_reaction.hpp`.
- [x] Verify C++ fixtures match TypeScript fixtures.
- [x] Remove or quarantine any binding code that references stale core paths.

## Phase 3 — Bind Cython / PufferLib

- [x] Add `training/setup.py` for building the Cython extension.
- [x] Replace stale `training/game.pyx` with `training/chain_reaction.pyx` including `core/chain_reaction.hpp`.
- [x] Expose reset, step, winner, legal-action mask, and observation methods through Cython.
- [x] Expose reset, step, winner, token readout, owner readout, and turn count through Cython.
- [x] Add a PufferLib v4 Ocean environment scaffold.
- [x] Verify the PufferLib v4 Ocean scaffold builds as a CPU backend on macOS.
- [x] Add a smoke test that instantiates one environment and validates reset, legal move, illegal move, and readout behavior.
- [x] Add a smoke test that runs random legal self-play episodes.

## Phase 4 — Train Model

- [x] Add PufferTank 4.0 Compose wiring for the Fedora CUDA workstation.
- [x] Build the PufferLib v4 Ocean environment on the CUDA workstation through PufferTank.
- [x] Run one tiny end-to-end PufferLib training job from reset to saved checkpoint.
- [x] Verify and pin the single-channel signed-distance-to-explosion observation tensor with corner, edge, center, empty, own, and opponent fixtures.
- [x] Add a repo-owned Torch PPO entrypoint that uses PufferLib Ocean only as the vectorized environment engine.
- [x] Implement the 32-channel residual CNN policy/value model.
- [x] Implement illegal-action masking before softmax.
- [x] Implement negamax/sign-aware GAE for alternating current-player-perspective turns.
- [x] Fork PufferLib v4 as a repo submodule (`vendor/PufferLib`, branch `chain-reaction-native`).
- [x] Publish fork to GitHub: `github.com/Nelathan/chain-reaction-pufferlib`, update `.gitmodules` and submodule remotes.
- [x] Hardcode Chain Reaction legal action mask (`obs >= 0`) in native CUDA `sample_logits`, `ppo_discrete_head`, and policy-gradient kernels.
- [x] Hardcode negamax alternating-turn GAE (`r - gamma*V(next)`, `-gamma*lambda*next_adv`) in native CUDA advantage kernels.
- [x] Review native CUDA submodule patch for sanity after DeepSeek implementation.
- [x] Fix native scalar/vector negamax advantage paths so horizon alignment cannot change TD-error math.
- [x] Rebuild patched PufferLib submodule inside the PufferTank image with `BUILD_ONLY=1`.
- [x] Run a tiny finite PufferTank training smoke against the patched submodule.
- [x] Verify legal mask at env level: `--log.illegal_moves` counter confirmed 0.000 in PufferTank training runs.
- [ ] Add a native CUDA/unit fixture proving scalar and vector negamax advantage paths match for `rho != 1` and multiple horizon alignments.
- [ ] Add an optional derived PufferTank Dockerfile for immutable CI/release builds after native CUDA semantics are fixture-verified.
- [x] Implement zero-sum self-play reward convention: log both winner (+1) and loser (-1) as separate PufferLib entries per game.
- [x] Add player-1-perspective winrate stat (`--log.winrate`) for skill tracking in the Ocean env dashboard.
- [x] Load a saved checkpoint and run an evaluation rollout.
- [x] Implement single-policy self-play.
- [x] Verify native training logs expose real terminal games, truncations, truncation rate, and mean episode length.
- [x] Run native telemetry smoke at `max_turns=64` and confirm current default training reaches only truncations under that cap.
- [x] Run native telemetry smoke at `max_turns=128` and confirm random/untrained rollout mostly reaches real terminal games under that cap.
- [x] Run matched-cap native train/eval checks at `max_turns=64` and record eval truncations plus winrate.
- [x] Evaluate the 64-turn-trained checkpoint at `max_turns=128` to measure cap sensitivity.
- [x] Run matched-cap native train/eval checks at `max_turns=128` and record eval truncations plus winrate.
- [ ] Add crafted terminal-credit fixtures for P1 win, P2 win, loser reward, zero truncation reward, and no terminal bootstrap.
- [ ] Decide whether 8x8 training needs a smaller-board curriculum before more long native runs.
- [ ] Implement native CUDA CNN policy/value model or explicitly demote native PufferLib training to environment-throughput smoke.
- [ ] Implement history-pool opponent sampling.
- [ ] Train a tiny CNN baseline beyond smoke-test scale.
- [x] Save model checkpoints.
- [x] Add rollout metrics for win rate, episode length, illegal-action rate, cascade depth, and truncation count.
- [x] Print native model architecture, parameter count, input shape, action head shape, and convolution presence at runtime.
- [x] Include checkpoint path in native JSON logs.
- [x] Confirm a native checkpoint can be loaded by PufferLib eval.
- [x] Add non-rendering checkpoint evaluation against random legal play.
- [ ] Expose exact per-game maximum cascade depth to the checkpoint evaluator.
- [ ] Inspect reward/value/entropy targets because checkpoint `0000000000262144.bin` only reached 58.4% vs random legal play and was trained with a shorter cap than evaluation.
- [ ] Fix `_C.close(policy)` segfault after direct native `forward_policy` evaluator use.
- [ ] Run a longer native single-policy training job only after terminal/truncation telemetry and train/eval caps are aligned.
- [ ] Evaluate multiple checkpoints from the same native run against random legal play only after cap alignment.
- [ ] Produce a checkpoint-progression table with steps, combined winrate, P1 winrate, P2 winrate, mean episode length, and truncations only after cap alignment.
- [x] Add a `pygame-ce` playable/debug shell before Godot integration.
- [ ] Export an inference artifact usable by Godot.

## Phase 5 — Bind Godot GDExtension

- [ ] Add `SConstruct` for Godot 4.x GDExtension builds.
- [ ] Rename or replace the current Godot source stub with the correct `.cpp` file.
- [ ] Update the Godot extension to include `core/chain_reaction.hpp`.
- [ ] Expose reset, action application, board readout, and winner readout to Godot.
- [ ] Render the board from the core state array.
- [ ] Add human input mapped to legal core actions.
- [ ] Add model inference with temperature scaling.
- [ ] Add a playable human-vs-AI scene.
