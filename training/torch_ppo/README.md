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
