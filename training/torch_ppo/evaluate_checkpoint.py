from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from training.torch_ppo.masking import apply_legal_mask, compute_cells_mask
from training.torch_ppo.model import ChainReactionNet
from training.torch_ppo.puffer_vec import PufferVec


BOARD_SIZE = 8  # fixed compile-time constant


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


def sample_random_legal(observations: torch.Tensor, generator: torch.Generator, valid_cells_mask: torch.Tensor | None = None) -> torch.Tensor:
    legal = observations.float() >= 0
    if valid_cells_mask is not None:
        legal = legal & valid_cells_mask.to(device=legal.device)
    if not bool(legal.any(dim=1).all().item()):
        raise RuntimeError("encountered observation with no legal actions")
    return torch.multinomial(legal.float(), 1, generator=generator).squeeze(1)


def sample_policy_actions(
    model: torch.nn.Module,
    observations: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    valid_cells_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        logits, _values = model(observations.float())
        masked_logits = apply_legal_mask(logits, observations.float(), valid_cells_mask)
        if temperature <= 0.0:
            return masked_logits.argmax(dim=1)
        return torch.multinomial(torch.softmax(masked_logits / temperature, dim=1), 1, generator=generator).squeeze(1)


def wilson_lower_bound(wins: int, games: int, z: float = 1.96) -> float | None:
    if games <= 0:
        return None
    phat = wins / games
    denom = 1.0 + z * z / games
    center = phat + z * z / (2.0 * games)
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * games)) / games)
    return (center - spread) / denom


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_checkpoint(path: Path, requested_board_size: int | None) -> tuple[dict, int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    checkpoint_board_size = int(config.get("board_size", 8))
    board_size = requested_board_size if requested_board_size is not None else checkpoint_board_size
    return checkpoint, checkpoint_board_size, board_size


def run_seat(
    args: argparse.Namespace,
    model: torch.nn.Module,
    checkpoint_seat: int,
    games: int,
    opponent_model: torch.nn.Module | None = None,
    valid_cells_mask: torch.Tensor | None = None,
) -> dict:
    if games <= 0:
        return {
            "games": 0,
            "wins": 0,
            "winrate": None,
            "truncations": 0,
            "illegal_moves": 0,
            "episode_lengths": [],
            "terminal_rewards": [],
            "env_log": {},
        }

    vec = PufferVec(
        load_puffer_args(
            args.total_agents, args.max_turns, args.seed + checkpoint_seat,
            getattr(args, "active_width", 8), getattr(args, "active_height", 8),
        ),
        sync_gpu_step=bool(args.sync_gpu_step),
    )
    expected_cells = args.board_size * args.board_size
    if vec.obs_size != expected_cells:
        vec.close()
        raise SystemExit(
            f"env obs_size {vec.obs_size} does not match board_size "
            f"{args.board_size} ({expected_cells} cells)"
        )

    generator = torch.Generator(device=vec.device)
    generator.manual_seed(args.seed + checkpoint_seat * 1000003)
    current_player = torch.ones(vec.total_agents, dtype=torch.int8, device=vec.device)
    episode_lengths = torch.zeros(vec.total_agents, dtype=torch.int32, device=vec.device)

    completed = 0
    wins = 0
    truncations = 0
    illegal_moves = 0
    completed_lengths: list[int] = []
    terminal_rewards: list[float] = []

    try:
        while completed < games:
            obs_snapshot = vec.observations.float().clone()
            acting_player = current_player.clone()
            actions = torch.empty(vec.total_agents, dtype=torch.long, device=vec.device)

            policy_turn = acting_player == checkpoint_seat
            if bool(policy_turn.any().item()):
                actions[policy_turn] = sample_policy_actions(
                    model,
                    obs_snapshot[policy_turn],
                    args.temperature,
                    generator,
                    valid_cells_mask,
                )
            if bool((~policy_turn).any().item()):
                if opponent_model is not None:
                    actions[~policy_turn] = sample_policy_actions(
                        opponent_model,
                        obs_snapshot[~policy_turn],
                        args.opponent_temperature,
                        generator,
                        valid_cells_mask,
                    )
                else:
                    actions[~policy_turn] = sample_random_legal(obs_snapshot[~policy_turn], generator, valid_cells_mask)

            chosen_legal = obs_snapshot.gather(1, actions.view(-1, 1)).squeeze(1) >= 0
            if not bool(chosen_legal.all().item()):
                illegal_moves += int((~chosen_legal).sum().item())
                bad = int(torch.nonzero(~chosen_legal, as_tuple=False).flatten()[0].item())
                raise RuntimeError(f"evaluator selected illegal action {int(actions[bad].item())} in env {bad}")

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
                    actor_was_checkpoint = actor == checkpoint_seat
                    checkpoint_won = (reward > 0.0 and actor_was_checkpoint) or (reward < 0.0 and not actor_was_checkpoint)
                    was_truncation = reward == 0.0 and int(length) >= args.max_turns
                    if checkpoint_won:
                        wins += 1
                    if was_truncation:
                        truncations += 1
                    completed_lengths.append(int(length))
                    terminal_rewards.append(float(reward if actor_was_checkpoint else -reward))
                    if completed >= games:
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

    return {
        "games": min(completed, games),
        "wins": wins,
        "winrate": wins / max(min(completed, games), 1),
        "truncations": truncations,
        "illegal_moves": illegal_moves,
        "episode_lengths": completed_lengths[:games],
        "terminal_rewards": terminal_rewards[:games],
        "env_log": env_log,
    }


def split_games(total: int, checkpoint_player: str) -> tuple[int, int]:
    if checkpoint_player == "p1":
        return total, 0
    if checkpoint_player == "p2":
        return 0, total
    return (total + 1) // 2, total // 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Torch PPO checkpoint against random legal play or another checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--total-agents", type=int, default=1024)
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE, help="model board size (always 8)")
    parser.add_argument("--active-width", type=int, default=8, help="active region width")
    parser.add_argument("--active-height", type=int, default=8, help="active region height")
    parser.add_argument("--max-turns", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--opponent-temperature", type=float, default=0.0)
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

    checkpoint, checkpoint_board_size, board_size = load_checkpoint(args.checkpoint, args.board_size)
    args.board_size = board_size
    active_w = args.active_width
    active_h = args.active_height
    valid_cells_mask = compute_cells_mask(BOARD_SIZE, active_w, active_h)
    if board_size != checkpoint_board_size:
        print(
            " ".join(
                [
                    "warning:",
                    f"evaluating checkpoint_board_size={checkpoint_board_size}",
                    f"at target_board_size={board_size}",
                    f"checkpoint={args.checkpoint}",
                ]
            ),
            flush=True,
        )
    model = ChainReactionNet(board_size=board_size)
    model.load_state_dict(checkpoint["model"])

    started = time.time()
    p1_games, p2_games = split_games(args.games, args.checkpoint_player)

    # Create the first vec before moving the model so the evaluator follows the
    # same CPU/GPU backend selected by PufferLib. The model is then reused across
    # both seat runs on that device.
    device_probe = PufferVec(load_puffer_args(1, args.max_turns, args.seed, active_w, active_h), sync_gpu_step=bool(args.sync_gpu_step))
    device = device_probe.device
    if device_probe.obs_size != board_size * board_size:
        actual = device_probe.obs_size
        device_probe.close()
        raise SystemExit(f"env obs_size {actual} does not match board_size {board_size}")
    device_probe.close()
    model = model.to(device).eval()
    valid_cells_mask = valid_cells_mask.to(device)

    opponent_model = None
    opponent_checkpoint_board_size: int | None = None
    if args.opponent_checkpoint is not None:
        if not args.opponent_checkpoint.exists():
            raise SystemExit(f"opponent checkpoint not found: {args.opponent_checkpoint}")
        opp_ckpt, opp_ckpt_board_size, _ = load_checkpoint(args.opponent_checkpoint, board_size)
        opponent_model = ChainReactionNet(board_size=board_size)
        opponent_model.load_state_dict(opp_ckpt["model"])
        opponent_model = opponent_model.to(device).eval()
        opponent_checkpoint_board_size = opp_ckpt_board_size

    p1 = run_seat(args, model, 1, p1_games, opponent_model, valid_cells_mask)
    p2 = run_seat(args, model, 2, p2_games, opponent_model, valid_cells_mask)

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

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_board_size": checkpoint_board_size,
        "target_board_size": board_size,
        "active_width": active_w,
        "active_height": active_h,
        "board_size": board_size,
        "obs_size": board_size * board_size,
        "games": total_games,
        "checkpoint_player": args.checkpoint_player,
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
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "seed": args.seed,
        "transfer_source_checkpoint": checkpoint.get("transfer_source_checkpoint"),
        "transfer_source_board_size": checkpoint.get("transfer_source_board_size"),
        "opponent_checkpoint": str(args.opponent_checkpoint) if args.opponent_checkpoint is not None else None,
        "opponent_checkpoint_board_size": opponent_checkpoint_board_size,
        "opponent_temperature": args.opponent_temperature if args.opponent_checkpoint is not None else None,
        "elapsed_seconds": time.time() - started,
    }

    if args.output is None:
        repo_root = Path(os.environ.get("CHAIN_REACTION_REPO", Path.cwd())).resolve()
        output_dir = repo_root / "training" / "evals" / "torch_ppo"
        output_dir.mkdir(parents=True, exist_ok=True)
        opponent_tag = "_vs_checkpoint" if args.opponent_checkpoint is not None else ""
        args.output = output_dir / f"eval_{checkpoint_board_size}x{checkpoint_board_size}_to_{board_size}x{board_size}{opponent_tag}_{int(started * 1000)}.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        " ".join(
            [
                f"games={report['games']}",
                f"checkpoint_board_size={report['checkpoint_board_size']}",
                f"target_board_size={report['target_board_size']}",
                f"checkpoint_player1_winrate={report['checkpoint_player1_winrate']}",
                f"checkpoint_player2_winrate={report['checkpoint_player2_winrate']}",
                f"combined_winrate={report['combined_winrate']}",
                f"wilson_lower_bound={report['wilson_lower_bound']}",
                f"illegal_selected_actions={report['illegal_selected_actions']}",
                f"truncations={report['truncations']}",
                f"mean_episode_length={report['mean_episode_length']}",
                f"output={args.output}",
            ]
        )
    )


if __name__ == "__main__":
    main()
