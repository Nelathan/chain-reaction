#!/usr/bin/env bash
set -euo pipefail

PUFFER_ROOT="${PUFFER_ROOT:-/puffertank/pufferlib}"
REPO_ROOT="${CHAIN_REACTION_REPO:-/workspace/chain-reaction}"
ENV_NAME="chain_reaction"
BOARD_SIZE="${CHAIN_REACTION_BOARD_SIZE:-8}"
INIT_CHECKPOINT="${CHAIN_REACTION_INIT_CHECKPOINT:-}"

cd "$PUFFER_ROOT"

if [ -f /puffertank/venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /puffertank/venv/bin/activate
fi

if [ ! -f "$REPO_ROOT/core/chain_reaction.hpp" ]; then
    echo "Missing shared core at $REPO_ROOT/core/chain_reaction.hpp" >&2
    exit 1
fi

mkdir -p ocean config checkpoints logs "$REPO_ROOT/training/checkpoints/torch_ppo" "$REPO_ROOT/training/logs/torch_ppo"
rm -rf "ocean/$ENV_NAME" "config/$ENV_NAME.ini" chain_reaction_core
ln -s "$REPO_ROOT/training/puffer_ocean/$ENV_NAME" "ocean/$ENV_NAME"
ln -s "$REPO_ROOT/training/puffer_ocean/config/$ENV_NAME.ini" "config/$ENV_NAME.ini"
ln -s "$REPO_ROOT/core" chain_reaction_core

# Compile the Ocean env at the requested board size.
EXTRA_CFLAGS="${EXTRA_CFLAGS:--I$REPO_ROOT} -DCR_WIDTH=$BOARD_SIZE -DCR_HEIGHT=$BOARD_SIZE" bash build.sh "$ENV_NAME" ${PUFFER_BUILD_ARGS:-}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
args=(
    python "$REPO_ROOT/training/torch_ppo/train.py"
    --total-timesteps "${CHAIN_REACTION_TOTAL_TIMESTEPS:-32768000}"
    --total-agents "${CHAIN_REACTION_TOTAL_AGENTS:-1024}"
    --horizon "${CHAIN_REACTION_HORIZON:-32}"
    --minibatch-size "${CHAIN_REACTION_MINIBATCH_SIZE:-32768}"
    --update-epochs "${CHAIN_REACTION_UPDATE_EPOCHS:-1}"
    --learning-rate "${CHAIN_REACTION_LEARNING_RATE:-0.0003}"
    --weight-decay "${CHAIN_REACTION_WEIGHT_DECAY:-0.0}"
    --gamma "${CHAIN_REACTION_GAMMA:-0.99}"
    --gae-lambda "${CHAIN_REACTION_GAE_LAMBDA:-0.95}"
    --clip-coef "${CHAIN_REACTION_CLIP_COEF:-0.2}"
    --vf-coef "${CHAIN_REACTION_VF_COEF:-0.5}"
    --ent-coef "${CHAIN_REACTION_ENT_COEF:-0.01}"
    --max-grad-norm "${CHAIN_REACTION_MAX_GRAD_NORM:-1.0}"
    --max-turns "${CHAIN_REACTION_MAX_TURNS:-136}"
    --board-size "$BOARD_SIZE"
    --seed "${CHAIN_REACTION_SEED:-73}"
    --checkpoint-interval "${CHAIN_REACTION_CHECKPOINT_INTERVAL:-100}"
    --log-interval "${CHAIN_REACTION_LOG_INTERVAL:-25}"
    --eval-interval "${CHAIN_REACTION_EVAL_INTERVAL:-100}"
    --eval-games "${CHAIN_REACTION_EVAL_GAMES:-32}"
    --compile-model "${CHAIN_REACTION_COMPILE_MODEL:--1}"
    --compile-mode "${CHAIN_REACTION_COMPILE_MODE:-default}"
    --sync-gpu-step "${CHAIN_REACTION_SYNC_GPU_STEP:-0}"
    --sync-timing "${CHAIN_REACTION_SYNC_TIMING:-0}"
    --wandb "${CHAIN_REACTION_WANDB:-0}"
    --wandb-project "${CHAIN_REACTION_WANDB_PROJECT:-chain-reaction}"
    --wandb-group "${CHAIN_REACTION_WANDB_GROUP:-torch-ppo}"
    --wandb-entity "${CHAIN_REACTION_WANDB_ENTITY:-}"
    --wandb-name "${CHAIN_REACTION_WANDB_NAME:-}"
    --wandb-tags "${CHAIN_REACTION_WANDB_TAGS:-}"
    --wandb-mode "${CHAIN_REACTION_WANDB_MODE:-}"
    --wandb-base-url "${CHAIN_REACTION_WANDB_BASE_URL:-}"
    --wandb-silent "${CHAIN_REACTION_WANDB_SILENT:-1}"
    --checkpoint-dir "$REPO_ROOT/training/checkpoints/torch_ppo"
    --log-dir "$REPO_ROOT/training/logs/torch_ppo"
)

if [ -n "$INIT_CHECKPOINT" ]; then
    args+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

"${args[@]}"
