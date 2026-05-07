from __future__ import annotations

import argparse
import json
import os
import time
from argparse import Namespace
from pathlib import Path

import torch

from training.torch_ppo.evaluate_checkpoint import (
    load_checkpoint,
    load_puffer_args,
    run_seat,
    split_games,
    wilson_lower_bound,
    percentile,
    BOARD_SIZE as EVAL_BOARD_SIZE,
)
from training.torch_ppo.masking import compute_cells_mask
from training.torch_ppo.model import ChainReactionNet
from training.torch_ppo.puffer_vec import PufferVec


def parse_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--board-sizes must contain at least one board size")
    return sizes


def evaluate_size(
    checkpoint: dict,
    checkpoint_path: Path,
    checkpoint_board_size: int,
    board_size: int,
    active_width: int,
    active_height: int,
    games: int,
    total_agents: int,
    max_turns: int,
    temperature: float,
    checkpoint_player: str,
    seed: int,
    sync_gpu_step: int,
) -> dict:
    args = Namespace(
        checkpoint=checkpoint_path,
        games=games,
        total_agents=total_agents,
        board_size=board_size,
        active_width=active_width,
        active_height=active_height,
        max_turns=max_turns,
        temperature=temperature,
        checkpoint_player=checkpoint_player,
        seed=seed,
        sync_gpu_step=sync_gpu_step,
    )

    model = ChainReactionNet(board_size=board_size)
    model.load_state_dict(checkpoint["model"])

    device_probe = PufferVec(load_puffer_args(1, max_turns, seed, active_width, active_height), sync_gpu_step=bool(sync_gpu_step))
    device = device_probe.device
    if device_probe.obs_size != board_size * board_size:
        actual = device_probe.obs_size
        device_probe.close()
        raise SystemExit(f"env obs_size {actual} does not match board_size {board_size}")
    device_probe.close()
    model = model.to(device).eval()
    valid_cells_mask = compute_cells_mask(EVAL_BOARD_SIZE, active_width, active_height).to(device)

    p1_games, p2_games = split_games(games, checkpoint_player)
    p1 = run_seat(args, model, 1, p1_games, valid_cells_mask=valid_cells_mask)
    p2 = run_seat(args, model, 2, p2_games, valid_cells_mask=valid_cells_mask)

    lengths = p1["episode_lengths"] + p2["episode_lengths"]
    terminal_rewards = p1["terminal_rewards"] + p2["terminal_rewards"]
    total_games = p1["games"] + p2["games"]
    total_wins = p1["wins"] + p2["wins"]
    total_truncations = p1["truncations"] + p2["truncations"]
    total_illegal = p1["illegal_moves"] + p2["illegal_moves"]
    combined_winrate = total_wins / total_games if total_games else None

    env_logs = [log for log in (p1.get("env_log"), p2.get("env_log")) if log]
    cascade_means = [float(log.get("mean_cascade_depth", log.get("cascade_depth", 0.0))) for log in env_logs]
    cascade_maxes = [float(log.get("max_cascade_depth", 0.0)) for log in env_logs]

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_board_size": checkpoint_board_size,
        "target_board_size": board_size,
        "board_size": board_size,
        "obs_size": board_size * board_size,
        "games": total_games,
        "checkpoint_player": checkpoint_player,
        "checkpoint_player1_games": p1["games"],
        "checkpoint_player1_wins": p1["wins"],
        "checkpoint_player1_winrate": p1["winrate"],
        "checkpoint_player2_games": p2["games"],
        "checkpoint_player2_wins": p2["wins"],
        "checkpoint_player2_winrate": p2["winrate"],
        "combined_winrate": combined_winrate,
        "wilson_lower_bound": wilson_lower_bound(total_wins, total_games),
        "illegal_selected_actions": total_illegal,
        "truncations": total_truncations,
        "terminal_rate": (total_games - total_truncations) / max(total_games, 1),
        "mean_episode_length": sum(lengths) / max(len(lengths), 1),
        "median_episode_length": percentile(lengths, 0.5),
        "p95_episode_length": percentile(lengths, 0.95),
        "p99_episode_length": percentile(lengths, 0.99),
        "mean_cascade_depth": sum(cascade_means) / max(len(cascade_means), 1) if cascade_means else None,
        "max_cascade_depth": max(cascade_maxes) if cascade_maxes else None,
        "mean_terminal_reward": sum(terminal_rewards) / max(len(terminal_rewards), 1),
        "max_turns": max_turns,
        "temperature": temperature,
        "seed": seed,
        "elapsed_seconds": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final multi-size evaluation for a Torch PPO checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--board-sizes", type=parse_sizes, default=parse_sizes("3,4,5,6,7,8"))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--total-agents", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--checkpoint-player", choices=("both", "p1", "p2"), default="both")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sync-gpu-step", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if args.total_agents <= 0:
        raise SystemExit("--total-agents must be positive")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    started = time.time()
    checkpoint, checkpoint_board_size, _target_board_size = load_checkpoint(args.checkpoint, None)
    size_reports = []
    for board_size in args.board_sizes:
        report = evaluate_size(
            checkpoint=checkpoint,
            checkpoint_path=args.checkpoint,
            checkpoint_board_size=checkpoint_board_size,
            board_size=board_size,
            active_width=board_size,
            active_height=board_size,
            games=args.games,
            total_agents=args.total_agents,
            max_turns=args.max_turns,
            temperature=args.temperature,
            checkpoint_player=args.checkpoint_player,
            seed=args.seed + board_size,
            sync_gpu_step=args.sync_gpu_step,
        )
        report["elapsed_seconds"] = time.time() - started
        size_reports.append(report)

    combined_winrates = [report["combined_winrate"] for report in size_reports if report["combined_winrate"] is not None]
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_board_size": checkpoint_board_size,
        "evaluated_board_sizes": args.board_sizes,
        "games_per_size": args.games,
        "total_agents": args.total_agents,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "checkpoint_player": args.checkpoint_player,
        "seed": args.seed,
        "size_reports": size_reports,
        "mean_combined_winrate": sum(combined_winrates) / max(len(combined_winrates), 1) if combined_winrates else None,
        "min_combined_winrate": min(combined_winrates) if combined_winrates else None,
        "elapsed_seconds": time.time() - started,
    }

    if args.output is None:
        repo_root = Path(os.environ.get("CHAIN_REACTION_REPO", Path.cwd())).resolve()
        output_dir = repo_root / "training" / "evals" / "torch_ppo"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"final_eval_{checkpoint_board_size}x{checkpoint_board_size}_{int(started * 1000)}.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        " ".join(
            [
                f"sizes={','.join(str(size) for size in args.board_sizes)}",
                f"mean_combined_winrate={report['mean_combined_winrate']}",
                f"min_combined_winrate={report['min_combined_winrate']}",
                f"output={args.output}",
            ]
        )
    )


if __name__ == "__main__":
    main()
