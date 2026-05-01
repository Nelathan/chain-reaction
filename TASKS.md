# TASKS.md

Atomic execution steps only. Each item should be binary: done or not done. No vague investigation tasks.

## Phase 1 — Finalize TypeScript Prototype

- [ ] Implement `ChainReaction.reset()` in `core/game.ts`.
- [ ] Implement legal move validation in `core/game.ts`.
- [ ] Implement `ChainReaction.step(actionIndex, playerId)` in `core/game.ts`.
- [ ] Implement double-buffered simultaneous cascade resolution in `core/game.ts`.
- [ ] Implement winner detection in `core/game.ts`.
- [ ] Ensure winner detection does not treat the first placed token as a win.
- [ ] Implement draw or max-turn handling in `core/game.ts`.
- [ ] Add TypeScript fixtures for corner critical mass.
- [ ] Add TypeScript fixtures for edge critical mass.
- [ ] Add TypeScript fixtures for center critical mass.
- [ ] Add TypeScript fixtures for simultaneous multi-cell explosions.
- [ ] Add TypeScript fixtures for ownership transfer during cascades.
- [ ] Add TypeScript fixtures for illegal moves.

## Phase 2 — Generate / Align C++ Header

- [ ] Replace `core/chain_reaction.hpp` stub with complete fixed-array implementation.
- [ ] Define board constants in `core/chain_reaction.hpp`.
- [ ] Define fixed-size token and owner arrays in `core/chain_reaction.hpp`.
- [ ] Implement critical mass calculation in `core/chain_reaction.hpp`.
- [ ] Implement legal move validation in `core/chain_reaction.hpp`.
- [ ] Implement double-buffered simultaneous cascade resolution in `core/chain_reaction.hpp`.
- [ ] Implement winner detection in `core/chain_reaction.hpp`.
- [ ] Implement signed-distance-to-explosion observation export in `core/chain_reaction.hpp`.
- [ ] Implement legal-action mask export in `core/chain_reaction.hpp`.
- [ ] Verify C++ fixtures match TypeScript fixtures.
- [ ] Remove or quarantine any binding code that references stale core paths.

## Phase 3 — Bind Cython / PufferLib

- [ ] Add `training/setup.py` for building the Cython extension.
- [ ] Update `training/game.pyx` to include `core/chain_reaction.hpp`.
- [ ] Expose reset, step, winner, legal-action mask, and observation methods through Cython.
- [ ] Implement a PufferLib environment wrapper.
- [ ] Add a smoke test that instantiates multiple environments.
- [ ] Add a smoke test that runs random legal self-play episodes.

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
