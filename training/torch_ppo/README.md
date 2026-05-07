# Chain Reaction Torch PPO

This directory holds the repo-owned trainer. PufferLib remains the fast Ocean environment engine; the policy, value model, masking, PPO update, checkpoints, and inspection code belong here so the learning loop stays readable.

Run it through the PufferTank container with the separate entrypoint:

```bash
CHAIN_REACTION_TOTAL_TIMESTEPS=10000000 \
CHAIN_REACTION_TOTAL_AGENTS=4096 \
CHAIN_REACTION_HORIZON=64 \
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer \
  bash /workspace/chain-reaction/training/torch_ppo/train.sh
```

The script builds the Ocean environment, then runs `training/torch_ppo/train.py` instead of PufferLib's native CUDA trainer.

To continue from a saved checkpoint on a new board size, set `CHAIN_REACTION_INIT_CHECKPOINT=/path/to/checkpoint.pt`. The model weights load, the optimizer starts fresh, and reports record both source and target board sizes.

For the post-training verdict across curriculum sizes `3x3` through `8x8`, use `training/torch_ppo/final_eval.py`.

To log the repo-owned trainer to Weights & Biases, export `WANDB_API_KEY` on the host and enable the trainer flag through Compose:

```bash
export WANDB_API_KEY=...

CHAIN_REACTION_WANDB=1 \
CHAIN_REACTION_WANDB_PROJECT=chain-reaction \
CHAIN_REACTION_WANDB_GROUP=torch-ppo \
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer \
  bash /workspace/chain-reaction/training/torch_ppo/train.sh
```

Optional pass-throughs are `CHAIN_REACTION_WANDB_ENTITY`, `CHAIN_REACTION_WANDB_NAME`, `CHAIN_REACTION_WANDB_TAGS`, `CHAIN_REACTION_WANDB_MODE`, and `CHAIN_REACTION_WANDB_BASE_URL`. W&B is disabled by default so smoke tests and offline runs do not require credentials. The trainer does not print per-update JSON; interval metrics are averaged and sent to W&B when enabled, while the full JSON log is still written at the end of the run.

Useful performance knobs:

- `CHAIN_REACTION_LOG_INTERVAL=10` logs every N PPO updates and averages interval metrics.
- `CHAIN_REACTION_COMPILE_MODEL=1` enables `torch.compile` for the policy/value model.
- `CHAIN_REACTION_COMPILE_MODE=default` forwards the compile mode to PyTorch.
- `CHAIN_REACTION_SYNC_GPU_STEP=0` keeps Puffer GPU stepping asynchronous; set to `1` only when debugging native step ordering.

## Training contract

- Observations are current-player-relative signed distance-to-explosion boards shaped as `(batch, 1, 8, 8)`.
- Empty cells are `0`.
- Current-player cells are positive: `critical_mass - tokens`.
- Opponent cells are negative: `-(critical_mass - tokens)`.
- Legal actions are exactly cells with observation value `>= 0`.
- Illegal actions must be masked on logits before constructing the categorical distribution. Do not softmax first and zero probabilities after.
- `V(s)` means value for the player to move. Since turns alternate, nonterminal bootstrapping uses `-V(s_next)` in GAE.

## Baseline model

The first model is deliberately tiny and spatial:

1. `Conv2d(1, 32, kernel_size=3, padding=1)` stem.
2. Four pre-activation residual blocks at constant 32 channels:
   `GroupNorm(8, 32) -> SiLU -> Conv3x3 -> GroupNorm(8, 32) -> SiLU -> Conv3x3 -> residual add`.
3. Policy head: `1x1` convolution to one logit per cell, then flatten to 64 logits and apply the legal mask.
4. Value head: `1x1` projection, global average pool, small MLP, scalar output. No `tanh` in the baseline.

The trunk does not stride, pool, or flatten. The board remains an 8x8 board until the policy head emits one score per cell.

## Why not native Puffer CUDA model work first?

PufferLib 4.0's native CUDA trainer is excellent for throughput, but custom model and masked-logit semantics live inside C++/CUDA internals. The goal here is learning RL and designing game-specific networks, not hiding the interesting parts behind a fast black box. If this Torch PPO path becomes too slow after it is correct and inspectable, native CUDA work can be justified with evidence.

## Recovery plan

The current primary path is to return to this trainer and stop iterating Chain Reaction model architecture inside the PufferLib submodule.

1. Use the exact baseline model in `model.py`; do not simplify it to match native Puffer constraints.
2. Run short PufferTank smokes first: finite losses, zero illegal sampled actions, and metrics written without relying on dashboard impressions.
3. Train with a game cap that reaches real terminals (`CHAIN_REACTION_MAX_TURNS >= 128` based on native telemetry); do not evaluate strength from all-truncation checkpoints.
4. Evaluate checkpoint progression against random legal play under the same cap used for training.
5. Add history-pool self-play only after the single-policy exact CNN shows measurable improvement.
6. Revisit native CUDA/Triton only with a concrete target: either accelerate a known-good Torch model or add a clean root-owned native model seam. Do not move architecture iteration back into the PufferLib fork by default.
