# Next Session Handoff

The curriculum experiment is a negative result and has been removed from the trainer. Keep the old checkpoints for diagnosis, but stop treating curriculum as an active path.

## Current Decision

- Train dedicated models per size for now.
- Keep transfer as an experiment, not the baseline.
- Revisit transfer only after we know whether the failure is representation, optimization, or a bug.

## Evidence So Far

- Curriculum checkpoint loses to scratch 8×8 in self-play.
- On 6×6 it wins cleanly.
- On 7×7 it shows seat asymmetry.
- Legality/truncation are fine; the problem is strength and consistency.
- The non-curriculum 8×8 checkpoint was healthy; keep that as the baseline reference.

## Future Targets To Keep Alive

These are still worth doing; they’re just no longer the main line:

- **Telemetry** — enable W&B and log losses, entropy, KL, and per-size evals.
- **PPO update speed** — if experiments get slow, target the update loop.

## Run Shape

Use the repo Torch path for evaluation and training, through PufferTank when native env wiring is needed.
Run both comparisons with the same hyperparameters and W&B enabled so the curves are directly comparable.

Current state:

- native `6x6` rerun is healthy
- native `4x4` run is healthy
- curriculum plumbing is removed
- no active-size runtime contract remains

Use these checkpoints for the next comparison:

```text
native 6x6:
  training/checkpoints/torch_ppo/1778274579436_0000000010092544.pt

native 4x4:
  training/checkpoints/torch_ppo/1778275483606_0000000010092544.pt
```

## Do Not Drift

- No claiming curriculum generalizes well.
- No reintroducing compile-time board-size overrides.
- No Godot work yet.
- No native PufferLib architecture detour until the size question is answered.
