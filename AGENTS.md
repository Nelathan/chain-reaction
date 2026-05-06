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
- Do not try to build or verify the native PufferLib CUDA trainer on the host. Use the PufferTank Compose path; the image is the toolchain contract.
- Do not optimize visuals before the AI/training loop exists.

## Native PufferLib Verification Protocol

The PufferLib fork lives at `vendor/PufferLib` on branch `chain-reaction-native`. It is intentionally single-purpose: Chain Reaction legal masking and negamax GAE are hardcoded in `src/pufferlib.cu` with no runtime flags.

When reviewing or changing the submodule:

1. Inspect the submodule diff directly in `vendor/PufferLib`.
2. Rebuild inside PufferTank, not on the host:

```bash
BUILD_ONLY=1 podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer
```

3. Run a tiny finite smoke after the build succeeds:

```bash
CHAIN_REACTION_TRAIN_TIMEOUT=3m \
CHAIN_REACTION_TOTAL_TIMESTEPS=8192 \
CHAIN_REACTION_CHECKPOINT_INTERVAL=1 \
CHAIN_REACTION_TOTAL_AGENTS=256 \
CHAIN_REACTION_MINIBATCH_SIZE=2048 \
CHAIN_REACTION_HORIZON=32 \
CHAIN_REACTION_MAX_TURNS=512 \
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer
```

4. Treat successful build plus tiny smoke as a smoke check only. It proves the image consumes the changed submodule and the loop breathes; it does not prove policy quality or kernel algebra.

The container build script creates untracked symlinks inside the submodule (`chain_reaction_core`, `config/chain_reaction.ini`, `ocean/chain_reaction`). Do not mistake them for source changes.

Known sharp edge: PufferLib has scalar and vector advantage kernels selected by horizon alignment. Any GAE change must keep both paths algebraically identical; otherwise `CHAIN_REACTION_HORIZON` silently changes learning math.

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
