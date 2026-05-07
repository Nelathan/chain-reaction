#!/usr/bin/env bash
set -euo pipefail

# Curriculum training: single Ocean compilation, round-robin batch scheduler.
# train.py handles all size switching internally — no process restarts.
# Run inside PufferTank only.

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

CHECKPOINT_DIR="$REPO_ROOT/training/checkpoints/torch_ppo"
LOG_DIR="$REPO_ROOT/training/logs/torch_ppo"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# --- One compilation for all sizes ---
mkdir -p ocean config
rm -rf "ocean/$ENV_NAME" "config/$ENV_NAME.ini" chain_reaction_core
ln -s "$REPO_ROOT/training/puffer_ocean/$ENV_NAME" "ocean/$ENV_NAME"
ln -s "$REPO_ROOT/training/puffer_ocean/config/$ENV_NAME.ini" "config/$ENV_NAME.ini"
ln -s "$REPO_ROOT/core" chain_reaction_core
EXTRA_CFLAGS="${EXTRA_CFLAGS:--I$REPO_ROOT}" bash build.sh "$ENV_NAME" ${PUFFER_BUILD_ARGS:-}
echo "built Ocean env at 8x8 (active region controlled at runtime)"

# --- Single training run, round-robin curriculum ---
python "$REPO_ROOT/training/torch_ppo/train.py" \
    --total-timesteps "${CHAIN_REACTION_TOTAL_TIMESTEPS:-32000000}" \
    --total-agents "${CHAIN_REACTION_TOTAL_AGENTS:-1024}" \
    --horizon "${CHAIN_REACTION_HORIZON:-32}" \
    --minibatch-size "${CHAIN_REACTION_MINIBATCH_SIZE:-8192}" \
    --unlock-interval "${CHAIN_REACTION_UNLOCK_INTERVAL:-100}" \
    --eval-interval "${CHAIN_REACTION_EVAL_INTERVAL:-100}" \
    --eval-games "${CHAIN_REACTION_EVAL_GAMES:-32}" \
    --sweep-interval "${CHAIN_REACTION_SWEEP_INTERVAL:-500}" \
    --sweep-games "${CHAIN_REACTION_SWEEP_GAMES:-64}" \
    --checkpoint-interval "${CHAIN_REACTION_CHECKPOINT_INTERVAL:-100}" \
    --init-checkpoint "${CHAIN_REACTION_INIT_CHECKPOINT:-}" \
    --compile-model "${CHAIN_REACTION_COMPILE_MODEL:-1}" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR"

echo "curriculum complete"
