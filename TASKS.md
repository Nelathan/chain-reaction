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
- [ ] Remove or quarantine any binding code that references stale core paths.

## Phase 3 — Bind Cython / PufferLib

- [x] Add `training/setup.py` for building the Cython extension.
- [x] Replace stale `training/game.pyx` with `training/chain_reaction.pyx` including `core/chain_reaction.hpp`.
- [x] Expose reset, step, winner, legal-action mask, and observation methods through Cython.
- [x] Expose reset, step, winner, token readout, owner readout, and turn count through Cython.
- [x] Implement a PufferLib environment wrapper.
- [x] Add a smoke test that instantiates multiple environments.
- [x] Add a smoke test that instantiates one environment and validates reset, legal move, illegal move, and readout behavior.
- [x] Add a smoke test that runs random legal self-play episodes.

## Phase 4 — Train Model

- [ ] Define the single-channel signed-distance-to-explosion observation tensor.
- [ ] Implement single-policy self-play.
- [ ] Implement history-pool opponent sampling.
- [ ] Implement illegal-action masking before softmax.
- [ ] Train a tiny CNN baseline.
- [ ] Save model checkpoints.
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
