# CHANGELOG.md

Chronological ledger. Record why changes happened, not just what changed.

## 2026-05-01

- Added root project governance files: `DESIGN.md`, `AGENTS.md`, `TASKS.md`, and `CHANGELOG.md`. This turns the frozen prototype skeleton into an executable contract: shared C++ core first, verified TypeScript semantics before bindings, and Godot/PufferLib as consumers instead of parallel rule engines.
- Established the double-buffered cascade rule as a design invariant. Chain Reaction explosions must resolve in simultaneous waves because in-place scan-order mutation can bias outcomes based on iteration direction rather than board state.
- Locked in flat fixed-size board state as a performance constraint. PufferLib needs thousands of duplicated environments per CPU core; allocator-heavy or pointer-heavy layouts would spend the budget on memory mechanics instead of environment stepping.
- Preserved the staged pipeline order: TypeScript prototype, C++ header verification, Cython/PufferLib binding, training, then Godot binding. The current repository already contains early consumer stubs, but binding work must wait until the core header is mathematically verified.
- Folded the GDD into the design contract: 8x8 deterministic Atoms mechanics, PPO self-play with a history pool, illegal-action masking, tiny 2D CNN policy/value network, and temperature-scaled Godot inference. This replaced the earlier three-channel observation placeholder with the sharper signed-distance-to-explosion channel so the model receives board physics directly.
