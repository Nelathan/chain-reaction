from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from training.torch_ppo.masking import compute_cells_mask
from training.torch_ppo.puffer_vec import PufferVec


BOARD_SIZE = 8


def sample_random_legal(observations: torch.Tensor, generator: torch.Generator, valid_cells_mask: torch.Tensor | None = None) -> torch.Tensor:
    legal = observations >= 0
    if valid_cells_mask is not None:
        legal = legal & valid_cells_mask.to(device=legal.device)
    if not bool(legal.any(dim=1).all().item()):
        raise RuntimeError("encountered observation with no legal actions")
    weights = legal.float()
    return torch.multinomial(weights, 1, generator=generator).squeeze(1)


def load_puffer_args(total_agents: int, max_turns: int, seed: int, active_width: int = 8, active_height: int = 8) -> dict:
    import pufferlib.pufferl

    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        args = pufferlib.pufferl.load_config("chain_reaction")
    finally:
        sys.argv = argv
    args["vec"]["total_agents"] = total_agents
    args["vec"]["num_buffers"] = 1
    args["env"]["max_turns"] = max_turns
    args["env"]["active_width"] = active_width
    args["env"]["active_height"] = active_height
    args["train"]["horizon"] = 1
    args["train"]["minibatch_size"] = max(1, total_agents)
    args["seed"] = seed
    args["reset_state"] = False
    args["cudagraphs"] = -1
    args["profile"] = False
    return args


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe random-vs-random Chain Reaction terminal lengths.")
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--active-width", type=int, default=8, help="active region width")
    parser.add_argument("--active-height", type=int, default=8, help="active region height")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--total-agents", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sync-gpu-step", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.board_size <= 0:
        raise SystemExit("--board-size must be positive")
    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.total_agents <= 0:
        raise SystemExit("--total-agents must be positive")

    started = time.time()
    vec = PufferVec(
        load_puffer_args(args.total_agents, args.max_turns, args.seed, args.active_width, args.active_height),
        sync_gpu_step=bool(args.sync_gpu_step),
    )
    expected_cells = args.board_size * args.board_size
    if vec.obs_size != expected_cells:
        vec.close()
        raise SystemExit(
            f"env obs_size {vec.obs_size} does not match board_size "
            f"{args.board_size} ({expected_cells} cells)"
        )

    valid_cells_mask = compute_cells_mask(BOARD_SIZE, args.active_width, args.active_height).to(vec.device)

    generator = torch.Generator(device=vec.device)
    generator.manual_seed(args.seed)
    current_player = torch.ones(vec.total_agents, dtype=torch.int8, device=vec.device)
    episode_lengths = torch.zeros(vec.total_agents, dtype=torch.int32, device=vec.device)

    completed = 0
    terminal_games = 0
    truncations = 0
    p1_wins = 0
    lengths: list[int] = []
    terminal_actor_rewards: list[float] = []

    try:
        while completed < args.games:
            acting_player = current_player.clone()
            actions = sample_random_legal(vec.observations.float(), generator, valid_cells_mask)
            vec.step(actions)
            episode_lengths += 1

            done = vec.terminals > 0.0
            done_idx = torch.nonzero(done, as_tuple=False).flatten()
            if done_idx.numel() > 0:
                rewards = vec.rewards[done_idx].float()
                actors = acting_player[done_idx]
                done_lengths = episode_lengths[done_idx]
                for reward, actor, length in zip(rewards.tolist(), actors.tolist(), done_lengths.tolist()):
                    completed += 1
                    lengths.append(int(length))
                    is_truncation = reward == 0.0 and int(length) >= args.max_turns
                    if is_truncation:
                        truncations += 1
                    else:
                        terminal_games += 1
                        terminal_actor_rewards.append(float(reward))
                        p1_won = (reward > 0.0 and actor == 1) or (reward < 0.0 and actor == 2)
                        if p1_won:
                            p1_wins += 1
                    if completed >= args.games:
                        break

                episode_lengths[done_idx] = 0
                current_player[done_idx] = 1

            still_running = ~done
            current_player[still_running] = torch.where(
                current_player[still_running] == 1,
                torch.tensor(2, dtype=torch.int8, device=vec.device),
                torch.tensor(1, dtype=torch.int8, device=vec.device),
            )
    finally:
        env_log = vec.log()
        vec.close()

    report = {
        "board_size": args.board_size,
        "active_width": args.active_width,
        "active_height": args.active_height,
        "obs_size": expected_cells,
        "requested_games": args.games,
        "games": min(completed, args.games),
        "max_turns": args.max_turns,
        "total_agents": args.total_agents,
        "terminal_games": terminal_games,
        "truncations": truncations,
        "terminal_rate": terminal_games / max(min(completed, args.games), 1),
        "truncation_rate": truncations / max(min(completed, args.games), 1),
        "p1_winrate": p1_wins / max(terminal_games, 1),
        "mean_episode_length": sum(lengths) / max(len(lengths), 1),
        "median_episode_length": percentile(lengths, 0.5),
        "p95_episode_length": percentile(lengths, 0.95),
        "p99_episode_length": percentile(lengths, 0.99),
        "p995_episode_length": percentile(lengths, 0.995),
        "max_episode_length": max(lengths) if lengths else None,
        "mean_terminal_actor_reward": sum(terminal_actor_rewards) / max(len(terminal_actor_rewards), 1),
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "env_log": env_log,
    }

    if args.output is None:
        repo_root = Path(os.environ.get("CHAIN_REACTION_REPO", Path.cwd())).resolve()
        output_dir = repo_root / "training" / "evals" / "torch_ppo"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"random_probe_{args.board_size}x{args.board_size}_{int(started * 1000)}.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        " ".join(
            [
                f"games={report['games']}",
                f"board_size={report['board_size']}",
                f"max_turns={report['max_turns']}",
                f"terminal_rate={report['terminal_rate']}",
                f"truncation_rate={report['truncation_rate']}",
                f"mean_episode_length={report['mean_episode_length']}",
                f"p99_episode_length={report['p99_episode_length']}",
                f"p995_episode_length={report['p995_episode_length']}",
                f"output={args.output}",
            ]
        )
    )


if __name__ == "__main__":
    main()
