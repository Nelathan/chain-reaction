# Chain Reaction

A high-throughput Chain Reaction / Atoms environment aimed at one playable, trainable prototype.

The project has one rule engine and several consumers. The C++ core owns runtime game behavior; Python, PufferLib, and Godot must expose or display that behavior, not reimplement it.

## Map

- `core/game.ts` — readable TypeScript prototype and rule fixtures.
- `core/chain_reaction.hpp` — zero-dependency C++ runtime truth.
- `training/` — Cython bridge, PufferLib Ocean environment, PPO experiments, checkpoint evaluation.
- `training/torch_ppo/` — repo-owned reference Torch PPO/CNN path.
- `training/puffer_ocean/` — native PufferLib Ocean consumer of the shared C++ core.
- `vendor/PufferLib` — forked PufferLib v4 submodule for Chain Reaction native CUDA patches.
- `src/` — future Godot 4.x GDExtension consumer.
- `DESIGN.md` — architecture, gameplay semantics, RL intent, and current design corrections.
- `TASKS.md` — binary execution checklist.
- `CHANGELOG.md` — chronological rationale and run history.

## Architecture contract

- Gameplay rules live in `core/chain_reaction.hpp`.
- Board state is fixed-size flat arrays; no heap allocation or STL containers in the core game loop.
- Board indexing is explicit: `index = y * WIDTH + x`.
- Cascades resolve as simultaneous waves through double-buffering.
- Training max-turn limits are harness truncations, not game rules.
- Godot is a renderer/input shell. PufferLib is a throughput harness.

If two consumers disagree, fix the core, the binding, or the test. Do not create a second rule engine.

## Quick start: CPU core checks

Use `uv`, not ambient Python. `training/setup.py` is currently cwd-sensitive.

```bash
cd training
uv run --group dev setup.py build_ext --inplace
uv run --group dev python test_bridge.py
uv run --group dev python test_random_self_play.py
```

## First playable shell

Build the Cython bridge, then run the pygame-ce shell from the repository root:

```bash
cd training
uv run --group dev setup.py build_ext --inplace
cd ..
uv run --group play python play.py
```

The pygame shell is presentation only: it renders core state, asks the core for legal moves, submits clicked actions, and displays the core-owned cascade log. It does not implement game rules.

To play against a Torch PPO checkpoint, keep Torch as the development runtime and pass a checkpoint path, or `latest` for the newest ignored `.pt` under the usual checkpoint/model directories:

```bash
uv run --group play --group ai python play.py --checkpoint latest --ai-player 2 --temperature 0.5
```

This is intentionally a convenience runtime. A tinygrad or native export can come later after the playable AI loop is worth slimming.

The first human-playable AI checkpoint came from a 30M-impression 8x8 Torch PPO run:

```bash
uv run --group play --group ai python play.py \
  --checkpoint training/checkpoints/torch_ppo/1778429882927_0000000030015488.pt \
  --ai-player 2 \
  --temperature 0.5
```

W&B logging uses `CHAIN_REACTION_WANDB_ENTITY`, falling back to `WANDB_ENTITY` through Compose.

## Torch PPO training shapes

The default Torch PPO launch shape is the 1k-update 8x8 profile:

```text
horizon=32
total_agents=1024
minibatch_size=32768
update_epochs=1
total_timesteps=32768000
```

Run it through PufferTank with the Torch entrypoint override:

```bash
podman compose -f compose.yaml -f compose.podman.yaml run --rm \
  -e CHAIN_REACTION_WANDB=1 \
  puffer bash -lc 'uv pip install --python /puffertank/venv/bin/python flashoptim==0.1.4 && bash /workspace/chain-reaction/training/torch_ppo/train.sh'
```

For the larger 10k-update profile, use the prepared launcher inside the same container path:

```bash
podman compose -f compose.yaml -f compose.podman.yaml run --rm \
  puffer bash -lc 'uv pip install --python /puffertank/venv/bin/python flashoptim==0.1.4 && bash /workspace/chain-reaction/training/torch_ppo/train_8x8_10k.sh'
```

The trainer logs active LR, pre-clip grad norm, PPO clip fraction, critic explained variance, and policy logit margin so loss/entropy/KL curves can be interpreted against optimizer pressure.

## PufferTank native path

Do not build or verify the native CUDA trainer on the host. Use the PufferTank image through Compose; the image is the toolchain contract.

From repo root, build the Chain Reaction Ocean env and patched native trainer:

```bash
BUILD_ONLY=1 \
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer
```

Run a small finite native smoke:

```bash
CHAIN_REACTION_TRAIN_TIMEOUT=3m \
CHAIN_REACTION_TOTAL_TIMESTEPS=32768 \
CHAIN_REACTION_CHECKPOINT_INTERVAL=1 \
CHAIN_REACTION_TOTAL_AGENTS=256 \
CHAIN_REACTION_MINIBATCH_SIZE=2048 \
CHAIN_REACTION_HORIZON=128 \
CHAIN_REACTION_MAX_TURNS=128 \
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer
```

The container script creates generated symlinks inside `vendor/PufferLib`:

- `chain_reaction_core`
- `config/chain_reaction.ini`
- `ocean/chain_reaction`

Those are build wiring, not source changes.

## Checkpoint evaluation

Use the finite evaluator, not PufferLib stock eval:

```bash
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer \
  bash /workspace/chain-reaction/training/puffer_ocean/puffertank_eval.sh \
    --checkpoint /workspace/chain-reaction/training/checkpoints/chain_reaction/<run>/<checkpoint>.bin \
    --games 1000 \
    --max-turns 128
```

`python -m pufferlib.pufferl eval chain_reaction --render-mode None` is not a strength evaluator. It loads weights, then enters an open-ended render/rollout loop even with rendering disabled.

For model-vs-model evaluation, run the evaluator inside PufferTank with the repo mounted at `/puffertank/pufferlib` and `PYTHONPATH=/puffertank/pufferlib:/puffertank/pufferlib/vendor/PufferLib`, then call `training/torch_ppo/evaluate_checkpoint.py --opponent-checkpoint ...`.

## Current training reality

The intended model is the small spatial CNN described in `DESIGN.md` and implemented in the repo Torch path. The current native PufferLib path is faster but currently trains PufferLib's default linear encoder + MinGRU + linear decoder. Runtime logs print this explicitly; trust the logs.

For 8x8 Chain Reaction, `max_turns=64` is too short: recent telemetry showed all games truncating at that cap. Use at least `max_turns=128` for meaningful native training/evaluation until a smaller-board curriculum or different cap is deliberately chosen.

Next useful work is not Godot polish or history-pool opponents. First verify terminal reward/value credit with crafted fixtures, then decide whether native PufferLib gets the intended CNN or remains an environment-throughput smoke path while repo-owned Torch PPO carries learning.

## More context

- Read `DESIGN.md` before changing architecture or training semantics.
- Use `TASKS.md` for the next binary task.
- Use `CHANGELOG.md` for run results and rationale.
