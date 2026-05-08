# Next Session Handoff

The curriculum experiment is now a negative result: the round-robin model is not the default path to strength. Keep it available for diagnosis, but stop treating it as the main line.

## Current Decision

- Train dedicated models per size for now.
- Keep curriculum transfer as an experiment, not the baseline.
- Revisit transfer only after we know whether the failure is representation, optimization, or a bug.

## Evidence So Far

- Curriculum checkpoint loses to scratch 8×8 in self-play.
- On 6×6 it wins cleanly.
- On 7×7 it shows seat asymmetry.
- Legality/truncation are fine; the problem is strength and consistency.

## Next Session Goal

Test whether the current 8×8-active implementation can match native size-specific training at smaller boards.

Recommended first comparison:

```text
train 6x6 directly
vs
train 8x8 with active_size=6
```

That is the next honest question.

## Future Targets To Keep Alive

These are still worth doing; they’re just no longer the main line:

- **Unlock-vs-flat curriculum ablation** — only if we want to isolate schedule effects.
- **AdamW audit** — make `weight_decay` explicit and decide on LR scheduling.
- **Telemetry** — enable W&B and log losses, entropy, KL, and per-size evals.
- **PPO update speed** — if experiments get slow, target the update loop.
- **6x6 transfer check** — compare native `6x6` training against `8x8` with `active_size=6`.

## Useful Checkpoints

```text
scratch 8x8:
  training/checkpoints/torch_ppo/1778149167954_0000000010092544.pt

curriculum 32M:
  training/checkpoints/torch_ppo/1778161398292_0000000032768000.pt
```

## Run Shape

Use the repo Torch path for evaluation and training, through PufferTank when native env wiring is needed.

## Do Not Drift

- No claiming curriculum generalizes well.
- No reintroducing compile-time board-size overrides.
- No Godot work yet.
- No native PufferLib architecture detour until the size question is answered.
