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
- Replaced turn-count winner gating with transition-based elimination state. The core now tracks seen and alive player masks, and a winner is produced only after an explosion leaves exactly one previously seen player alive.
- Added core-owned legal-action and signed-distance observation exports. This keeps training consumers from reimplementing legality, critical mass, or perspective math in Python.
