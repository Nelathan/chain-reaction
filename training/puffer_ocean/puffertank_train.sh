#!/usr/bin/env bash
set -euo pipefail

PUFFER_ROOT="${PUFFER_ROOT:-/puffertank/pufferlib}"
REPO_ROOT="${CHAIN_REACTION_REPO:-/workspace/chain-reaction}"
ENV_NAME="chain_reaction"

cd "$PUFFER_ROOT"

if [ -f /puffertank/venv/bin/activate ]; then
    # PufferTank ships PufferLib and its CUDA Python deps in this environment.
    # shellcheck disable=SC1091
    source /puffertank/venv/bin/activate
fi

if [ ! -f "$REPO_ROOT/core/chain_reaction.hpp" ]; then
    echo "Missing shared core at $REPO_ROOT/core/chain_reaction.hpp" >&2
    exit 1
fi

mkdir -p ocean config checkpoints logs
rm -rf "ocean/$ENV_NAME" "config/$ENV_NAME.ini" chain_reaction_core
ln -s "$REPO_ROOT/training/puffer_ocean/$ENV_NAME" "ocean/$ENV_NAME"
ln -s "$REPO_ROOT/training/puffer_ocean/config/$ENV_NAME.ini" "config/$ENV_NAME.ini"
ln -s "$REPO_ROOT/core" chain_reaction_core

echo "Using PufferLib at $PUFFER_ROOT"
echo "Using Chain Reaction repo at $REPO_ROOT"
echo "Building Ocean environment: $ENV_NAME"

EXTRA_CFLAGS="${EXTRA_CFLAGS:--I$REPO_ROOT}" bash build.sh "$ENV_NAME" ${PUFFER_BUILD_ARGS:-}

if [ "${BUILD_ONLY:-0}" = "1" ]; then
    echo "BUILD_ONLY=1; skipping training."
    exit 0
fi

TRAIN_TIMEOUT="${CHAIN_REACTION_TRAIN_TIMEOUT:-10m}"
TOTAL_TIMESTEPS="${CHAIN_REACTION_TOTAL_TIMESTEPS:-4000000000}"
CHECKPOINT_INTERVAL="${CHAIN_REACTION_CHECKPOINT_INTERVAL:-200}"
TOTAL_AGENTS="${CHAIN_REACTION_TOTAL_AGENTS:-4096}"
MINIBATCH_SIZE="${CHAIN_REACTION_MINIBATCH_SIZE:-32768}"
HORIZON="${CHAIN_REACTION_HORIZON:-64}"
MAX_TURNS="${CHAIN_REACTION_MAX_TURNS:-4096}"

echo "Training for up to $TRAIN_TIMEOUT; target timesteps=$TOTAL_TIMESTEPS"

set +e
timeout --signal=INT --kill-after=30s "$TRAIN_TIMEOUT" \
    python -m pufferlib.pufferl train "$ENV_NAME" \
        --train.total-timesteps "$TOTAL_TIMESTEPS" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --vec.total-agents "$TOTAL_AGENTS" \
        --train.minibatch-size "$MINIBATCH_SIZE" \
        --train.horizon "$HORIZON" \
        --env.max-turns "$MAX_TURNS"
status=$?
set -e

if [ "$status" -eq 124 ] || [ "$status" -eq 130 ]; then
    echo "Training stopped by ${TRAIN_TIMEOUT} timeout. Checkpoints/logs remain mounted under training/."
    exit 0
fi

exit "$status"
