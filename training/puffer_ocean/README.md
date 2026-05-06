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

The preferred first real training target is the CUDA workstation using PufferTank Docker / Podman with this repo's PufferLib submodule fork.

## Native trainer fidelity contract

This repository vendors PufferLib v4 as a git submodule at `vendor/PufferLib` with a
`chain-reaction-native` branch carrying two semantic patches to `src/pufferlib.cu`:

- **Legal action masking**: illegal (opponent-owned) cells are masked out of rollout
  sampling, PPO logprob/entropy, and policy-gradient calculation. The mask is derived
  from the current-player-relative observation: `obs >= 0`.
- **Negamax GAE**: `V(s)` is from the player-to-move perspective, so nonterminal
  advantage bootstraps with sign flips: `r - gamma * V(next)` and
  `- gamma * lambda * next_advantage`.

There are no runtime conditionals; this fork is single-purpose for Chain Reaction.
The Torch reference model at `training/torch_ppo/model.py` and fixtures remain as
readable reference until the native trainer is verified against them.

## Self-play reward convention

The Ocean env logs two terminal entries per completed game — winner (+1) and
loser (-1) — so that both sides receive explicit zero-sum rewards. Without the
loser entry, the losing player's final move has reward 0 (game ongoing), and
the value function must infer the loss entirely through bootstrap. The double
entry means `episode_return` mean = 0 and `perf` = 0.5 for balanced play; track
actual skill via the `winrate` stat (player 1 wins / games from player 1's
perspective).

The illegal-move penalty (-1 for the fouling player, +1 for the opponent)
follows the same zero-sum pattern and is guarded against by the CUDA legal
action mask (`obs >= 0`).  In practice this counter should always read 0.

## PufferTank + submodule build

The PufferLib submodule (`vendor/PufferLib`, branch `chain-reaction-native`) points
at a public fork: `git@github.com:Nelathan/chain-reaction-pufferlib.git`.

```bash
# From repo root:
git submodule init
git submodule update
cd vendor/PufferLib && git checkout chain-reaction-native && cd ../..

# Podman:
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer

# Docker:
docker compose -f compose.yaml -f compose.docker.yaml run --rm puffer

# Build-only smoke (skips training):
BUILD_ONLY=1 podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer

# Finite checkpoint evaluation against random legal play:
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer \
  bash /workspace/chain-reaction/training/puffer_ocean/puffertank_eval.sh \
    --checkpoint /workspace/chain-reaction/training/checkpoints/chain_reaction/1778090124759/0000000000262144.bin \
    --games 1000 \
    --max-turns 512
```

The compose mounts the repo at `/workspace/chain-reaction` inside the container,
uses the submodule as `PUFFER_ROOT`, and runs `puffertank_train.sh` which builds
the environment and launches training via `python -m pufferlib.pufferl`.

Do not regress to PyPI `pufferlib==3.0.0` for convenience. That package exposes a different integration API and would rot the v4 contract.

Do not use `python -m pufferlib.pufferl eval chain_reaction --render-mode None` for checkpoint strength evaluation. The stock PufferLib eval function loops forever (`render`, then `rollouts`) even when rendering is disabled; use the finite evaluator script above so the run ends with JSON metrics instead of becoming a CPU space heater.

Latest cap-aligned result: a `max_turns=128`, horizon `128`, 1,048,576-step native smoke produced checkpoint `training/checkpoints/chain_reaction/1778096039119/0000000001572864.bin`; matched `128`-cap evaluation scored `0.579` combined winrate over `1000` deterministic games vs random legal play (`P1=0.564`, `P2=0.594`, illegal moves `0`, truncations `15`, mean episode length `120.303`). The JSON report is `training/evals/chain_reaction/cap128_1778096039119_1572864.json`.

Caveat: the earlier `0.584` result for checkpoint `1778090124759/0000000000262144.bin` was evaluated at `max_turns=512` after training at `max_turns=64`. Matched `64`-cap evaluation gives `1000/1000` truncations, so that checkpoint is not clean strength evidence.

## Local Cython bridge smoke

The repo-local Cython bridge is a CPU verification path for `core/chain_reaction.hpp`. It does not require CUDA, PufferTank, or a PufferLib checkout. On Fedora, install the C++ compiler driver once:

```bash
sudo dnf install -y gcc-c++
```

Then build and test through `uv` from the `training/` directory. `training/setup.py` currently expects that working directory because its extension source is declared as `chain_reaction.pyx`.

```bash
uv run --group dev python setup.py build_ext --inplace
uv run --group dev python test_bridge.py
```

If `uv run --group dev python training/setup.py build_ext --inplace` is run from the repository root, Cython will fail with `chain_reaction.pyx doesn't match any files`. That is a cwd issue, not a core or compiler failure.

## Fedora CUDA workstation path

The official Puffer docs point at PufferTank Docker. The important contract is in PufferTank's `docker.sh`, not in cloning that repo into this project:

- image: `pufferai/puffertank:4.0`;
- GPU access: Docker `--gpus all`;
- process/runtime shape: host IPC, host cgroup namespace, host networking;
- interactive/rendering support: X11 socket, `$DISPLAY`, `$XAUTHORITY`, Wayland/audio envs;
- working tree inside the container: `/puffertank/pufferlib` with PufferLib already installed in `/puffertank/venv`.

This repository mirrors that runtime shape with Compose, then adds only the Chain Reaction-specific mount/symlink step. No Puffertank subrepo is required. The container script symlinks the Ocean env, config, and core include directory into PufferLib, so `core/chain_reaction.hpp` remains shared instead of copied.

### Docker path

Fedora does not ship `nvidia-container-toolkit` from the base repos. Add NVIDIA's RPM repo first, per NVIDIA's container-toolkit docs:

```bash
sudo dnf install -y curl
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Fedora 44's ca-certificates package owns this bundle path. If NVIDIA's repo
# file points at missing /etc/pki/tls/certs/ca-bundle.crt, dnf will fail with
# Curl error 77 before it can read repo metadata.
sudo sed -i \
  's#^sslcacert=.*#sslcacert=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem#' \
  /etc/yum.repos.d/nvidia-container-toolkit.repo

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1
sudo dnf install -y \
  nvidia-container-toolkit-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1-${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then run Chain Reaction through this repo's Compose wrapper:

```bash
docker compose -f compose.yaml -f compose.docker.yaml run --rm puffer
```

### Podman path

Podman can work, but it is not byte-for-byte Puffer's documented runner because PufferTank's `docker.sh` is Docker-specific. If avoiding Docker remains important, use NVIDIA CDI and keep `NVIDIA_VISIBLE_DEVICES` out of the Podman run:

```bash
sudo dnf install -y curl podman-compose
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo sed -i \
  's#^sslcacert=.*#sslcacert=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem#' \
  /etc/yum.repos.d/nvidia-container-toolkit.repo

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1
sudo dnf install -y \
  nvidia-container-toolkit-base-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1-${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
podman run --rm --device nvidia.com/gpu=all docker.io/pufferai/puffertank:4.0 nvidia-smi

podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer
```

Useful knobs:

```bash
CHAIN_REACTION_TRAIN_TIMEOUT=10m \
CHAIN_REACTION_TOTAL_TIMESTEPS=4000000000 \
CHAIN_REACTION_TOTAL_AGENTS=4096 \
docker compose -f compose.yaml -f compose.docker.yaml run --rm puffer
```

Checkpoints and logs are mounted to `training/checkpoints/` and `training/logs/`. If the shell timeout stops training before `total_timesteps`, the latest interval checkpoint is still the artifact to inspect. This is a product probe, not proof of strength: ten minutes is enough to reveal whether the loop breathes and whether the opponent has texture, not enough to certify balanced play.
