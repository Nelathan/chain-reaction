# AGENTS.md

You are maintaining a high-performance Chain Reaction environment. Core logic resides in C++. You are forbidden from using `std::vector`, `std::map`, or any heap allocations in the core game loop. Use flat 1D arrays. Output C-style C++. Do not touch Godot or Python bindings until the core header is mathematically verified.

## Mission

Preserve the shared-core architecture and move the project toward a playable, trainable prototype.

The repository has four layers:

1. `core/game.ts` — human-readable TypeScript rules prototype.
2. `core/chain_reaction.hpp` — single-header C++ runtime truth.
3. `training/` — Python / Cython / PufferLib consumer.
4. `src/` — Godot 4.x GDExtension consumer.

Only the C++ core owns runtime game behavior. Bindings expose it. Renderers display it. Training harnesses step it.

## Hard Rules

- Do not duplicate gameplay rules in Godot or Python.
- Do not introduce heap allocation into the core game loop.
- Do not use `std::vector`, `std::map`, `std::unordered_map`, dynamic polymorphism, exceptions, or dependency-bearing STL containers in the core.
- Use fixed-size flat 1D arrays for board state.
- Keep board indexing explicit: `index = y * WIDTH + x`.
- Preserve simultaneous explosion semantics through double-buffering.
- Keep the C++ header zero-dependency beyond fixed-width C/C++ primitive headers.
- Do not touch Godot or Python bindings until the C++ core is mathematically verified against the TypeScript prototype.
- Use `uv` for Python dependency management, environments, builds, and test execution. Do not use ambient `python`, `pip`, or global site packages for repo work unless the user explicitly authorizes an exception.
- Do not optimize visuals before the AI/training loop exists.

## Verification Order

1. Finalize and test the TypeScript prototype.
2. Implement the C++ header with equivalent behavior.
3. Verify C++ against TypeScript fixtures and edge cases.
4. Add Cython/PufferLib bindings.
5. Add Godot GDExtension bindings.

Changing this order requires a strong reason recorded in `CHANGELOG.md`.

## Core Semantics To Protect

- Legal moves may target empty cells or cells owned by the acting player.
- Illegal moves must be rejected deterministically.
- Cascades must resolve as simultaneous waves, not scan-order mutation.
- Ownership transfers happen when tokens are received during explosions.
- Terminal-state logic must distinguish early-game transient single-player ownership from actual elimination.
- All consumers must observe the same board state representation.

## Style

- Prefer boring, explicit C-style C++.
- Prefer named constants over magic numbers.
- Prefer small pure helpers over clever abstractions.
- Add comments only where they protect non-obvious semantics, especially simultaneous explosion behavior.
- Record architectural decisions in `DESIGN.md` and chronological rationale in `CHANGELOG.md`.

## Current Bias

Ship the smallest honest version that can be verified, trained, and rendered. No ornamental machinery. No shrine to future complexity. The board explodes; the policy learns; Godot watches.
