# PufferLib v4 Ocean scaffold

PufferLib v4 trains native Ocean environments from a PufferLib source checkout. It does not expose the legacy Python `pufferlib.emulation` / `PufferEnv` wrapper API.

To test this scaffold locally on macOS CPU:

1. Clone PufferLib 4.0 outside this repo.
2. Copy or symlink `training/puffer_ocean/chain_reaction` to `PufferLib/ocean/chain_reaction`.
3. Copy or symlink `training/puffer_ocean/config/chain_reaction.ini` to `PufferLib/config/chain_reaction.ini`.
4. Build with an include path back to this repository root so `core/chain_reaction.hpp` is shared, not duplicated:

```bash
EXTRA_CFLAGS="-I/path/to/chain-reaction" bash build.sh chain_reaction --cpu
```

On the current M1 setup, PufferLib's stock macOS `build.sh` needed local compiler flag tweaks for Homebrew GCC/OpenMP. Those tweaks belong upstream or in local setup notes, not in this repository's game core.

The preferred first real training target is the CUDA workstation using PufferTank Docker or a native PufferLib 4.0 checkout. The trainer and environment run in the same process against the compiled `pufferlib._C` module; there is no HTTP boundary between trainer and environment in the normal PufferLib v4 path. Mount or copy this repository into the container/workspace, expose this Ocean env inside the PufferLib checkout, build, then train there.

Do not regress to PyPI `pufferlib==3.0.0` for convenience. That package exposes a different integration API and would rot the v4 contract.
