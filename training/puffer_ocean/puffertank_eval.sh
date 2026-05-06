#!/usr/bin/env bash
set -euo pipefail

PUFFER_ROOT="${PUFFER_ROOT:-/puffertank/pufferlib}"
REPO_ROOT="${CHAIN_REACTION_REPO:-/workspace/chain-reaction}"
ENV_NAME="chain_reaction"

cd "$PUFFER_ROOT"
export PYTHONPATH="$PUFFER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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
echo "Building Ocean environment for checkpoint evaluation: $ENV_NAME"

EXTRA_CFLAGS="${EXTRA_CFLAGS:--I$REPO_ROOT}" bash build.sh "$ENV_NAME" ${PUFFER_BUILD_ARGS:-}

exec python "$REPO_ROOT/training/puffer_ocean/evaluate_checkpoint.py" "$@"
