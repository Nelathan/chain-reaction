#!/usr/bin/env python3
"""Evaluate a native Chain Reaction checkpoint against random legal play."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from pufferlib import _C
from pufferlib import pufferl as pufferl_cli


ACTION_COUNT = 64


def _cpu_float_view(ptr: int, shape: tuple[int, ...]) -> np.ndarray:
    n = math.prod(shape)
    buf = (ctypes.c_float * n).from_address(ptr)
    return np.ctypeslib.as_array(buf).reshape(shape)


def _masked_policy_actions(logits: np.ndarray, obs: np.ndarray, temperature: float, rng: np.random.Generator) -> np.ndarray:
    action_logits = logits[:, :ACTION_COUNT].astype(np.float64, copy=True)
    legal = obs[:, :ACTION_COUNT] >= 0.0
    action_logits[~legal] = -np.inf

    if np.any(~np.isfinite(action_logits).all(axis=1) & ~legal.any(axis=1)):
        raise RuntimeError("encountered observation with no legal actions")

    if temperature <= 0.0:
        return np.argmax(action_logits, axis=1).astype(np.int32)

    actions = np.empty(action_logits.shape[0], dtype=np.int32)
    scaled = action_logits / float(temperature)
    for i, row in enumerate(scaled):
        finite = np.isfinite(row)
        shifted = row[finite] - np.max(row[finite])
        probs = np.exp(shifted)
        probs /= probs.sum()
        legal_actions = np.flatnonzero(finite)
        actions[i] = int(rng.choice(legal_actions, p=probs))
    return actions


def _random_legal_action(obs_row: np.ndarray, rng: np.random.Generator) -> int:
    legal_actions = np.flatnonzero(obs_row[:ACTION_COUNT] >= 0.0)
    if legal_actions.size == 0:
        raise RuntimeError("encountered observation with no legal actions")
    return int(rng.choice(legal_actions))


def _prepare_args(games: int, max_turns: int, seed: int) -> dict:
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        args = pufferl_cli.load_config("chain_reaction")
    finally:
        sys.argv = saved_argv
    args["vec"]["total_agents"] = games
    args["vec"]["num_buffers"] = 1
    args["env"]["max_turns"] = max_turns
    args["train"]["horizon"] = 1
    args["train"]["minibatch_size"] = max(1, games)
    args["seed"] = seed
    args["reset_state"] = False
    args["cudagraphs"] = -1
    args["profile"] = False
    return args


def _run_seat(args: argparse.Namespace, checkpoint_seat: int, games: int) -> dict:
    if games <= 0:
        return {
            "games": 0,
            "wins": 0,
            "winrate": None,
            "truncations": 0,
            "illegal_moves": 0,
            "episode_lengths": [],
            "terminal_rewards": [],
        }

    rng = np.random.default_rng(args.seed + checkpoint_seat * 1000003)
    cfg = _prepare_args(games, args.max_turns, args.seed + checkpoint_seat)

    policy = _C.create_pufferl(cfg)
    _C.load_weights(policy, str(args.checkpoint))

    vec = _C.create_vec(cfg, 0)
    vec.reset()
    obs = _cpu_float_view(vec.obs_ptr, (games, vec.obs_size))
    rewards = _cpu_float_view(vec.rewards_ptr, (games,))
    terminals = _cpu_float_view(vec.terminals_ptr, (games,))
    actions = np.zeros((games, vec.num_atns), dtype=np.float32)

    current_player = np.ones(games, dtype=np.int8)
    episode_lengths = np.zeros(games, dtype=np.int32)
    completed = 0
    wins = 0
    truncations = 0
    illegal_moves = 0
    finished = np.zeros(games, dtype=bool)
    completed_lengths: list[int] = []
    terminal_rewards: list[float] = []

    try:
        while completed < games:
            active = np.flatnonzero(~finished)
            obs_snapshot = obs.copy()
            acting_player = current_player.copy()
            checkpoint_turn = acting_player[active] == checkpoint_seat

            actions[:, 0] = 0.0
            policy_idx = active[checkpoint_turn]
            if policy_idx.size:
                logits = _C.forward_policy(policy, np.ascontiguousarray(obs_snapshot[policy_idx], dtype=np.float32))
                policy_actions = _masked_policy_actions(logits, obs_snapshot[policy_idx], args.temperature, rng)
                actions[policy_idx, 0] = policy_actions.astype(np.float32)

            random_idx = active[~checkpoint_turn]
            for env_i in random_idx:
                actions[env_i, 0] = float(_random_legal_action(obs_snapshot[env_i], rng))
            for env_i in np.flatnonzero(finished):
                actions[env_i, 0] = float(_random_legal_action(obs_snapshot[env_i], rng))

            chosen = actions[:, 0].astype(np.int32)
            legal = obs_snapshot[active, chosen[active]] >= 0.0
            if not np.all(legal):
                illegal_moves += int((~legal).sum())
                bad = int(active[np.flatnonzero(~legal)[0]])
                raise RuntimeError(f"evaluator selected illegal action {chosen[bad]} in env {bad}")

            vec.cpu_step(actions.ctypes.data)
            episode_lengths[active] += 1

            done_idx = np.flatnonzero((terminals > 0.0) & ~finished)
            for env_i in done_idx:
                reward = float(rewards[env_i])
                actor_was_checkpoint = acting_player[env_i] == checkpoint_seat
                checkpoint_won = (reward > 0.0 and actor_was_checkpoint) or (reward < 0.0 and not actor_was_checkpoint)
                was_truncation = reward == 0.0 and episode_lengths[env_i] >= args.max_turns
                if checkpoint_won:
                    wins += 1
                if was_truncation:
                    truncations += 1
                completed += 1
                completed_lengths.append(int(episode_lengths[env_i]))
                terminal_rewards.append(reward if actor_was_checkpoint else -reward)
                finished[env_i] = True
                episode_lengths[env_i] = 0
                current_player[env_i] = 1

            still_active = active[terminals[active] <= 0.0]
            current_player[still_active] = np.where(current_player[still_active] == 1, 2, 1)
    finally:
        if hasattr(vec, "chain_reaction_raw_log"):
            env_log = dict(vec.chain_reaction_raw_log())
        else:
            env_log = dict(vec.log())
        vec.close()
        # `_C.close(policy)` currently segfaults after direct `forward_policy`
        # use because native PuffeRL worker-thread teardown assumes the stock
        # rollout path. The process is short-lived, so let OS cleanup own this.

    return {
        "games": completed,
        "wins": wins,
        "winrate": wins / completed if completed else None,
        "truncations": truncations,
        "illegal_moves": illegal_moves,
        "episode_lengths": completed_lengths,
        "terminal_rewards": terminal_rewards,
        "env_log": env_log,
    }


def _split_games(total: int, checkpoint_player: str) -> tuple[int, int]:
    if checkpoint_player == "p1":
        return total, 0
    if checkpoint_player == "p2":
        return 0, total
    return (total + 1) // 2, total // 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--max-turns", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--checkpoint-player", choices=("both", "p1", "p2"), default="both")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.games <= 0:
        raise SystemExit("--games must be positive")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    started = time.time()
    p1_games, p2_games = _split_games(args.games, args.checkpoint_player)
    p1 = _run_seat(args, 1, p1_games)
    p2 = _run_seat(args, 2, p2_games)

    lengths = p1["episode_lengths"] + p2["episode_lengths"]
    terminal_rewards = p1["terminal_rewards"] + p2["terminal_rewards"]
    total_games = p1["games"] + p2["games"]
    total_wins = p1["wins"] + p2["wins"]
    env_logs = [log for log in (p1.get("env_log"), p2.get("env_log")) if log]
    cascade_means = [float(log.get("mean_cascade_depth", log.get("cascade_depth", 0.0))) for log in env_logs]

    report = {
        "checkpoint": str(args.checkpoint),
        "games": total_games,
        "checkpoint_player": args.checkpoint_player,
        "checkpoint_player1_games": p1["games"],
        "checkpoint_player1_wins": p1["wins"],
        "checkpoint_player1_winrate": p1["winrate"],
        "checkpoint_player2_games": p2["games"],
        "checkpoint_player2_wins": p2["wins"],
        "checkpoint_player2_winrate": p2["winrate"],
        "combined_winrate": total_wins / total_games if total_games else None,
        "illegal_moves": p1["illegal_moves"] + p2["illegal_moves"],
        "truncations": p1["truncations"] + p2["truncations"],
        "mean_episode_length": float(np.mean(lengths)) if lengths else None,
        "median_episode_length": float(np.median(lengths)) if lengths else None,
        "mean_cascade_depth": float(np.mean(cascade_means)) if cascade_means else None,
        "max_cascade_depth": None,
        "max_cascade_depth_note": "Exact per-game max is not exposed by the current PufferLib aggregate log.",
        "mean_terminal_reward": float(np.mean(terminal_rewards)) if terminal_rewards else None,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
    }

    if args.output is None:
        repo_root = Path(os.environ.get("CHAIN_REACTION_REPO", Path.cwd())).resolve()
        output_dir = repo_root / "training" / "evals" / "chain_reaction"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"{int(started * 1000)}.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        " ".join(
            [
                f"games={report['games']}",
                f"checkpoint_player1_winrate={report['checkpoint_player1_winrate']}",
                f"checkpoint_player2_winrate={report['checkpoint_player2_winrate']}",
                f"combined_winrate={report['combined_winrate']}",
                f"mean_episode_length={report['mean_episode_length']}",
                f"illegal_moves={report['illegal_moves']}",
                f"truncations={report['truncations']}",
                f"mean_cascade_depth={report['mean_cascade_depth']}",
                f"max_cascade_depth={report['max_cascade_depth']}",
                f"output={args.output}",
            ]
        )
    )


if __name__ == "__main__":
    main()
