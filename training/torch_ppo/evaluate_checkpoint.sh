#!/usr/bin/env bash
set -euo pipefail

PUFFER_ROOT="${PUFFER_ROOT:-/puffertank/pufferlib}"
REPO_ROOT="${CHAIN_REACTION_REPO:-/workspace/chain-reaction}"
ENV_NAME="chain_reaction"
BOARD_SIZE="${CHAIN_REACTION_BOARD_SIZE:-8}"
CHECKPOINT="${CHAIN_REACTION_CHECKPOINT:-}"

if [ -z "$CHECKPOINT" ]; then
    echo "CHAIN_REACTION_CHECKPOINT is required" >&2
    exit 1
fi

cd "$PUFFER_ROOT"

if [ -f /puffertank/venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source /puffertank/venv/bin/activate
fi

if [ ! -f "$REPO_ROOT/core/chain_reaction.hpp" ]; then
    echo "Missing shared core at $REPO_ROOT/core/chain_reaction.hpp" >&2
    exit 1
fi

mkdir -p ocean config "$REPO_ROOT/training/evals/torch_ppo"
rm -rf "ocean/$ENV_NAME" "config/$ENV_NAME.ini" chain_reaction_core
ln -s "$REPO_ROOT/training/puffer_ocean/$ENV_NAME" "ocean/$ENV_NAME"
ln -s "$REPO_ROOT/training/puffer_ocean/config/$ENV_NAME.ini" "config/$ENV_NAME.ini"
ln -s "$REPO_ROOT/core" chain_reaction_core

EXTRA_CFLAGS="${EXTRA_CFLAGS:--I$REPO_ROOT} -DCR_WIDTH=$BOARD_SIZE -DCR_HEIGHT=$BOARD_SIZE" bash build.sh "$ENV_NAME" ${PUFFER_BUILD_ARGS:-}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
python "$REPO_ROOT/training/torch_ppo/evaluate_checkpoint.py" \
    --checkpoint "$CHECKPOINT" \
    --board-size "$BOARD_SIZE" \
    --games "${CHAIN_REACTION_EVAL_GAMES:-1000}" \
    --total-agents "${CHAIN_REACTION_TOTAL_AGENTS:-1024}" \
    --max-turns "${CHAIN_REACTION_MAX_TURNS:-512}" \
    --temperature "${CHAIN_REACTION_TEMPERATURE:-0.0}" \
    --checkpoint-player "${CHAIN_REACTION_CHECKPOINT_PLAYER:-both}" \
    --seed "${CHAIN_REACTION_SEED:-1}"
