#!/usr/bin/env bash
set -euo pipefail

PUFFER_ROOT="${PUFFER_ROOT:-/puffertank/pufferlib}"
REPO_ROOT="${CHAIN_REACTION_REPO:-/workspace/chain-reaction}"
ENV_NAME="chain_reaction"

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

EXTRA_CFLAGS="${EXTRA_CFLAGS:--I$REPO_ROOT}" bash build.sh "$ENV_NAME" ${PUFFER_BUILD_ARGS:-}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
python "$REPO_ROOT/training/torch_ppo/train.py" \
    --total-timesteps "${CHAIN_REACTION_TOTAL_TIMESTEPS:-10000000}" \
    --total-agents "${CHAIN_REACTION_TOTAL_AGENTS:-4096}" \
    --horizon "${CHAIN_REACTION_HORIZON:-64}" \
    --minibatch-size "${CHAIN_REACTION_MINIBATCH_SIZE:-32768}" \
    --max-turns "${CHAIN_REACTION_MAX_TURNS:-4096}" \
    --checkpoint-interval "${CHAIN_REACTION_CHECKPOINT_INTERVAL:-20}" \
    --checkpoint-dir "$REPO_ROOT/training/checkpoints/torch_ppo" \
    --log-dir "$REPO_ROOT/training/logs/torch_ppo"
