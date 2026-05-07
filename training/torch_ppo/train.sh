#!/usr/bin/env bash
set -euo pipefail

PUFFER_ROOT="${PUFFER_ROOT:-/puffertank/pufferlib}"
REPO_ROOT="${CHAIN_REACTION_REPO:-/workspace/chain-reaction}"
ENV_NAME="chain_reaction"
ACTIVE_WIDTH="${CHAIN_REACTION_ACTIVE_WIDTH:-8}"
ACTIVE_HEIGHT="${CHAIN_REACTION_ACTIVE_HEIGHT:-8}"
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

# Always compile at 8x8 (the model's fixed board size). The active region is
# controlled by active_width/active_height at runtime, not compile-time constants.
EXTRA_CFLAGS="${EXTRA_CFLAGS:--I$REPO_ROOT}" bash build.sh "$ENV_NAME" ${PUFFER_BUILD_ARGS:-}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
args=(
    python "$REPO_ROOT/training/torch_ppo/train.py"
    --total-timesteps "${CHAIN_REACTION_TOTAL_TIMESTEPS:-10000000}"
    --total-agents "${CHAIN_REACTION_TOTAL_AGENTS:-1024}"
    --horizon "${CHAIN_REACTION_HORIZON:-128}"
    --minibatch-size "${CHAIN_REACTION_MINIBATCH_SIZE:-8192}"
    --max-turns "${CHAIN_REACTION_MAX_TURNS:-128}"
    --active-width "$ACTIVE_WIDTH"
    --active-height "$ACTIVE_HEIGHT"
    --checkpoint-interval "${CHAIN_REACTION_CHECKPOINT_INTERVAL:-20}"
    --checkpoint-dir "$REPO_ROOT/training/checkpoints/torch_ppo"
    --log-dir "$REPO_ROOT/training/logs/torch_ppo"
)

if [ -n "$INIT_CHECKPOINT" ]; then
    args+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

"${args[@]}"
