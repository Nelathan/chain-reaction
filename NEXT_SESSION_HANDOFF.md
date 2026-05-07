# Next Session Handoff

The round-robin batch curriculum is implemented and verified. `train.py` handles all board sizes in one session — no process restarts, no bash-level size loops. One Ocean compilation, one `train.py` invocation.

## Run Command

Inside PufferTank container (from repo root):

```bash
CHAIN_REACTION_TOTAL_TIMESTEPS=32768000 \
CHAIN_REACTION_HORIZON=32 \
CHAIN_REACTION_CHECKPOINT_INTERVAL=100 \
CHAIN_REACTION_TOTAL_AGENTS=1024 \
CHAIN_REACTION_MINIBATCH_SIZE=8192 \
podman compose -f compose.yaml -f compose.podman.yaml run --rm puffer \
  bash /workspace/chain-reaction/training/torch_ppo/curriculum.sh
```

The compose default command runs the native PufferLib trainer (`puffertank_train.sh`). Override with `bash .../curriculum.sh`.
Env vars must precede `podman compose` on the same command line — `-e` flags after `run --rm` also work.

Key knobs:
- `CHAIN_REACTION_HORIZON`: 32 (smaller rollout, faster updates)
- `CHAIN_REACTION_UNLOCK_INTERVAL`: 100 (new size joins round-robin every 100 updates)
- `CHAIN_REACTION_TOTAL_TIMESTEPS`: 32768000 (= 32M, ~1000 updates at 1024 agents × 32 horizon)
- `CHAIN_REACTION_EVAL_INTERVAL`: 100 (evaluate largest unlocked size)
- `CHAIN_REACTION_SWEEP_INTERVAL`: 500 (evaluate all unlocked sizes)

## Current Good Checkpoints

```text
4x4 dedicated (99.8% against random):
  training/checkpoints/torch_ppo/1778140129666_0000000010027008.pt

8x8 scratch (100% against random, dominated transfer 8x8 100%-0%):
  training/checkpoints/torch_ppo/1778149167954_0000000010092544.pt

Round-robin curriculum 1000-update (100% all sizes against random):
  training/checkpoints/torch_ppo/1778161398292_0000000032768000.pt
```

## Round-Robin Curriculum Design

- **Scheduler**: `CurriculumScheduler` in `train.py`. Sizes [4, 5, 6, 7, 8]. Unlocks at update 0, 100, 200, 300, 400. Round-robin cycling after unlock.
- **Env switching**: PufferVec destroyed/recreated on size change (~10ms overhead vs ~400ms per update). No env spamming in logs.
- **Eval**: every 100 updates against the LARGEST unlocked size (tracking hardest progress). Sweep eval every 500 across all unlocked sizes. Final sweep at end.
- **Per-size caps**: hardcoded in `SIZE_CAPS = {4: 32, 5: 64, 6: 80, 7: 104, 8: 136}`.

## Verified Results (32M impressions, ~1000 updates)

100% winrate against random legal play on all sizes 4×4–8×8. Wilson lower bound 0.943 (64-game samples). Zero illegal actions, zero truncations.

Random-legal-play is dead as a benchmark. The 100% across all sizes proves the model generalizes, but skill ceiling is unknown.

## Next Concrete Cuts

### Cut 1: Clean up deprecated env/size config

`CHAIN_REACTION_ACTIVE_WIDTH`, `CHAIN_REACTION_ACTIVE_HEIGHT`, `CHAIN_REACTION_BOARD_SIZE` in compose.yaml and train.sh are vestigial — board is always 8×8 with active mask. The scheduler controls sizes now. Remove stale env vars and CLI args.

### Cut 2: Self-play evaluation

The curriculum model scores 100% against random on all sizes. That's a floor, not a ceiling. Evaluate against the scratch 8×8 model (`1778149167954_0000000010092544.pt`) using `evaluate_checkpoint.py --opponent-checkpoint`.

### Cut 3: Longer training / slower unlocks

1000 updates worked, but more updates per size may produce stronger agents. Try:
- `CHAIN_REACTION_TOTAL_TIMESTEPS=65536000` (2000 updates)
- `CHAIN_REACTION_UNLOCK_INTERVAL=200` (slower progression, more time on small boards)

### Cut 4: PPO Update Loop Speed

At 32-horizon, rollout is ~60ms and PPO update is ~400ms. Still 7× dominated by update. Triton kernel fusion is the likely lever.

## Do Not Drift

- Do not reintroduce compile-time board size overrides. Mask-only.
- Do not add eval-based gating back. Step-based unlocks are simpler and proven.
- Do not evaluate against random legal play as the sole benchmark.
- Do not touch Godot yet.
- Do not switch back to native PufferLib model work.
