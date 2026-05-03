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
