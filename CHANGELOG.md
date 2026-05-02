# CHANGELOG.md

Chronological ledger. Record why changes happened, not just what changed.

## 2026-05-01

- Added root project governance files: `DESIGN.md`, `AGENTS.md`, `TASKS.md`, and `CHANGELOG.md`. This turns the frozen prototype skeleton into an executable contract: shared C++ core first, verified TypeScript semantics before bindings, and Godot/PufferLib as consumers instead of parallel rule engines.
- Established the double-buffered cascade rule as a design invariant. Chain Reaction explosions must resolve in simultaneous waves because in-place scan-order mutation can bias outcomes based on iteration direction rather than board state.
- Locked in flat fixed-size board state as a performance constraint. PufferLib needs thousands of duplicated environments per CPU core; allocator-heavy or pointer-heavy layouts would spend the budget on memory mechanics instead of environment stepping.
- Preserved the staged pipeline order: TypeScript prototype, C++ header verification, Cython/PufferLib binding, training, then Godot binding. The current repository already contains early consumer stubs, but binding work must wait until the core header is mathematically verified.
- Folded the GDD into the design contract: 8x8 deterministic Atoms mechanics, PPO self-play with a history pool, illegal-action masking, tiny 2D CNN policy/value network, and temperature-scaled Godot inference. This replaced the earlier three-channel observation placeholder with the sharper signed-distance-to-explosion channel so the model receives board physics directly.
- Implemented matching TypeScript and C++ core rules for legal moves, critical mass, deterministic double-buffered cascades, ownership transfer, and winner detection. The provided implementation sketch was adjusted to preserve the opening transient: the first placed token is not a win in a two-player elimination game.
- Added a TypeScript/C++ fixture equivalence harness. This gives the project a cheap regression gate before Cython or Godot bindings are allowed to consume the core.
- Recorded `uv` as the mandatory Python toolchain. Training and Cython work must not depend on ambient Python, pip, or global site packages because reproducibility matters before speed claims mean anything.
- Added the first Cython bridge after the TypeScript/C++ equivalence gate passed. The bridge exposes the verified C++ header to Python without duplicating gameplay rules, and it is built/tested through `uv` so training work does not depend on ambient interpreter state.
- Replaced the stale `training/game.pyx` consumer that referenced `core-logic/chain_reaction.hpp`. Keeping a broken path around after the shared core moved to `core/chain_reaction.hpp` would invite accidental parallel plumbing.
- Replaced scan-order ownership overwrite with simultaneous pressure cancellation. Same-owner pressure stacks; opposing two-player pressure cancels to a net value, so engine iteration order no longer decides contested cells.
- Replaced turn-count winner gating with transition-based elimination state. This was first modeled with seen and alive masks, then simplified to alive-only once the game model settled on all participants starting alive and eliminations occurring only after explosion transitions.
- Added core-owned legal-action and signed-distance observation exports. This keeps training consumers from reimplementing legality, critical mass, or perspective math in Python.
- Simplified player lifecycle state to an alive mask initialized with all participating players. Winner is now derived from that mask rather than stored, and eliminations are recomputed only after explosion transitions.
- Clarified that simulation state and training observation are separate schemas. The core stores owner plus token count because transient overfull cells are valid during cascade resolution; signed-distance observation is only a derived stable-state view.
- Accepted unbounded cascade resolution for the MVP. Any future wave cap must be an explicit bounded-mode/error-state design, not a quiet physics change.
- Added explicit corner, edge, and center critical-mass fixtures. The MVP keeps the 8x8 chessboard shape, and these tests pin the geometry before renderer or training consumers build assumptions on top of it.
- Added a fixed-size per-move wave log to the shared core and Cython bridge. Rendering can now animate actual simultaneous cascade waves without re-simulating rules, while simulation remains unbounded; only the inspection log can truncate.

## 2026-05-02

- Rejected core draw/max-turn handling. Chain Reaction gameplay remains unbounded; training may use a high episode cap as harness-level truncation only, so batch management cannot silently become physics.
- Marked the corner, edge, and center critical-mass fixture tasks complete now that the latest verification work includes them.
- Briefly prototyped, then removed, a legacy PufferLib-style Python wrapper. PufferLib v4 does not expose that API, so keeping it would have advertised the wrong training seam.
- Clarified that turns are alternating while cascade waves are simultaneous. Opposing-pressure cancellation fixtures remain as mathematical symmetry checks for the resolver, even when ordinary stable-turn play rarely reaches those arbitrary wave states.
- Added a random legal self-play smoke test against the Cython core. The test samples only from core-owned legal masks and expects episodes to reach a real winner before the smoke-test harness cap.
- Split the Cython bridge into logged and fast stepping paths. Rendering/debug consumers can keep `step()` wave logs, while training uses `step_fast()` so rollout throughput does not pay for cascade inspection buffers.
- Changed cascade terminal semantics to stop after any simultaneous wave that leaves exactly one player with tokens. A legal random self-play rollout exposed a period-26 owner-only cascade after P2 had already been eliminated; winner detection now terminates the turn at elimination instead of requiring post-victory physics to stabilize.
- Replaced the provisional Python `PufferEnv` wrapper with a PufferLib v4 Ocean-style native environment scaffold. PufferLib v4 does not expose the old `pufferlib.emulation`/`PufferEnv` API; the trainable path is `ocean/<env>/binding.c` built with `build.sh chain_reaction --cpu` on macOS or the CUDA backend on the workstation.
- Chose workstation-first training as the next milestone. The M1 laptop remains a verification/smoke environment; the next useful product loop is a minimal CUDA/PufferTank training run, then a CLI/TUI policy replay viewer, then Godot once the policy loop proves life.
