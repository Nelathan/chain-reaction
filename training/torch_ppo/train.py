from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from dataclasses import asdict, dataclass

import torch
from torch import nn

from training.torch_ppo.gae import compute_negamax_gae
from training.torch_ppo.masking import compute_cells_mask, masked_categorical
from training.torch_ppo.model import ChainReactionNet
from training.torch_ppo.puffer_vec import PufferVec
from training.torch_ppo.evaluate_checkpoint import (
    percentile as eval_percentile,
    run_seat as eval_run_seat,
    split_games as eval_split_games,
    wilson_lower_bound as eval_wilson_lower_bound,
)


BOARD_SIZE = 8  # fixed compile-time constant; all models are 8×8 CNN

# per-size max_turns caps from random-vs-random probes
SIZE_CAPS = {3: 16, 4: 32, 5: 64, 6: 80, 7: 104, 8: 136}


@dataclass
class TrainConfig:
    total_timesteps: int = 32_000_000
    total_agents: int = 1024
    horizon: int = 32
    minibatch_size: int = 8192
    update_epochs: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    max_turns: int = 128
    init_checkpoint: str = ""
    seed: int = 73
    checkpoint_interval: int = 100
    checkpoint_dir: str = "training/checkpoints/torch_ppo"
    log_dir: str = "training/logs/torch_ppo"
    wandb: int = 0
    wandb_project: str = "chain-reaction"
    wandb_group: str = "torch-ppo"
    wandb_entity: str = ""
    wandb_name: str = ""
    wandb_tags: str = ""
    wandb_mode: str = ""
    wandb_base_url: str = ""
    wandb_silent: int = 1
    log_interval: int = 25
    eval_interval: int = 100
    eval_games: int = 32
    sweep_interval: int = 500
    sweep_games: int = 64
    unlock_interval: int = 100
    compile_model: int = 1
    compile_mode: str = "default"
    sync_gpu_step: int = 0


class CurriculumScheduler:
    """Round-robin batch scheduler. Unlocks sizes at step thresholds, then cycles."""

    SIZES = [4, 5, 6, 7, 8]

    def __init__(self, unlock_interval: int = 100):
        self.unlock_interval = unlock_interval

    def get_size(self, update: int) -> int:
        """Return the board size to use for this update."""
        n = min(len(self.SIZES), update // self.unlock_interval + 1)
        return self.SIZES[update % n]

    def unlocked_sizes(self, update: int) -> list[int]:
        return self.SIZES[: min(len(self.SIZES), update // self.unlock_interval + 1)]

    def newly_unlocked(self, update: int) -> list[int]:
        prev = self.unlocked_sizes(update - 1) if update > 0 else []
        curr = self.unlocked_sizes(update)
        return [s for s in curr if s not in prev]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Chain Reaction with repo-owned Torch PPO")
    defaults = TrainConfig()
    for field, value in asdict(defaults).items():
        arg = "--" + field.replace("_", "-")
        env_value = os.environ.get("CHAIN_REACTION_" + field.upper())
        default = type(value)(env_value) if env_value is not None else value
        parser.add_argument(arg, type=type(value), default=default)
    ns = parser.parse_args()
    return TrainConfig(**vars(ns))


def load_puffer_args(config: TrainConfig, active_size: int) -> dict:
    import pufferlib.pufferl

    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        args = pufferlib.pufferl.load_config("chain_reaction")
    finally:
        sys.argv = argv
    args["vec"]["total_agents"] = config.total_agents
    args["vec"]["num_buffers"] = 1
    args["env"]["max_turns"] = SIZE_CAPS.get(active_size, 128)
    args["env"]["active_width"] = active_size
    args["env"]["active_height"] = active_size
    args["train"]["horizon"] = config.horizon
    args["train"]["minibatch_size"] = config.minibatch_size
    args["train"]["total_timesteps"] = config.total_timesteps
    return args


def create_vec(active_size: int, config: TrainConfig, sync_gpu: bool) -> PufferVec:
    return PufferVec(load_puffer_args(config, active_size), sync_gpu_step=sync_gpu)


def evaluate_policy(model: nn.Module, obs: torch.Tensor, valid_cells_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = model(obs)
    dist = masked_categorical(logits, obs, valid_cells_mask)
    actions = dist.sample()
    logprobs = dist.log_prob(actions)
    return actions, logprobs, values


def flatten_log(log: dict, prefix: str = "") -> dict[str, float | int | str]:
    flat: dict[str, float | int | str] = {}
    for key, value in log.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_log(value, name))
        elif isinstance(value, (float, int, str)):
            flat[name] = value
    return flat


class IntervalAverager:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.tensors: dict[str, list[torch.Tensor]] = {}

    def add(self, values: dict[str, float | int | torch.Tensor]) -> None:
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                self.tensors.setdefault(key, []).append(value.detach())
            elif isinstance(value, (float, int)):
                self.totals[key] = self.totals.get(key, 0.0) + float(value)
                self.counts[key] = self.counts.get(key, 0) + 1

    def pop(self) -> dict[str, float]:
        averaged = {
            key: self.totals[key] / max(self.counts.get(key, 0), 1)
            for key in self.totals
        }
        for key, values in self.tensors.items():
            averaged[key] = float(torch.stack(values).mean().item())
        self.totals.clear()
        self.counts.clear()
        self.tensors.clear()
        return averaged


def init_wandb(config: TrainConfig, run_id: str):
    if config.wandb == 0:
        return None
    if config.wandb_mode:
        os.environ["WANDB_MODE"] = config.wandb_mode
    if config.wandb_base_url:
        os.environ["WANDB_BASE_URL"] = config.wandb_base_url
    if config.wandb_silent:
        os.environ["WANDB_SILENT"] = "true"
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("CHAIN_REACTION_WANDB=1 requires wandb in the training environment") from exc

    tags = [tag.strip() for tag in config.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        group=config.wandb_group or None,
        name=config.wandb_name or run_id,
        id=run_id,
        config=asdict(config),
        tags=tags or None,
    )


def maybe_compile_model(model: nn.Module, config: TrainConfig) -> nn.Module:
    if config.compile_model == 0:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("CHAIN_REACTION_COMPILE_MODEL=1 requires torch.compile")
    return torch.compile(model, mode=config.compile_mode)


def validate_env_shape(vec: PufferVec) -> None:
    expected_cells = BOARD_SIZE * BOARD_SIZE
    if vec.obs_size != expected_cells:
        raise RuntimeError(
            f"env obs_size {vec.obs_size} does not match board_size "
            f"{BOARD_SIZE} ({expected_cells} cells); rebuild the Ocean env "
            "with matching CR_WIDTH/CR_HEIGHT"
        )


def load_init_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer | None, config: TrainConfig) -> dict[str, int | str | None] | None:
    if not config.init_checkpoint:
        return None

    checkpoint_path = config.init_checkpoint
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"init checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint_config = checkpoint.get("config", {})
    source_active_w = checkpoint_config.get("active_width", BOARD_SIZE)
    source_active_h = checkpoint_config.get("active_height", BOARD_SIZE)
    return {
        "source_checkpoint": checkpoint_path,
        "source_active_width": source_active_w,
        "source_active_height": source_active_h,
    }


def run_eval(
    model: nn.Module,
    active_size: int,
    config: TrainConfig,
    update: int,
    n_games: int,
) -> dict[str, float | int | None]:
    eval_agents = min(config.total_agents, 256)
    max_turns = SIZE_CAPS.get(active_size, 128)
    mask = compute_cells_mask(BOARD_SIZE, active_size, active_size)
    eval_args = Namespace(
        total_agents=eval_agents,
        max_turns=max_turns,
        seed=config.seed + update * 1000003,
        board_size=BOARD_SIZE,
        active_width=active_size,
        active_height=active_size,
        temperature=0.0,
        checkpoint_player="both",
        sync_gpu_step=config.sync_gpu_step,
    )
    p1_games, p2_games = eval_split_games(n_games, eval_args.checkpoint_player)
    mask = mask.to(next(model.parameters()).device)
    p1 = eval_run_seat(eval_args, model, 1, p1_games, valid_cells_mask=mask)
    p2 = eval_run_seat(eval_args, model, 2, p2_games, valid_cells_mask=mask)
    total_games = p1["games"] + p2["games"]
    total_wins = p1["wins"] + p2["wins"]
    total_truncations = p1["truncations"] + p2["truncations"]
    total_illegal = p1["illegal_moves"] + p2["illegal_moves"]
    lengths = p1["episode_lengths"] + p2["episode_lengths"]
    return {
        "active_size": active_size,
        "games": total_games,
        "combined_winrate": total_wins / total_games if total_games else None,
        "p1_winrate": p1["winrate"],
        "p2_winrate": p2["winrate"],
        "wilson_lower_bound": eval_wilson_lower_bound(total_wins, total_games),
        "illegal_selected_actions": total_illegal,
        "truncations": total_truncations,
        "terminal_rate": (total_games - total_truncations) / max(total_games, 1),
        "mean_episode_length": sum(lengths) / max(len(lengths), 1),
        "median_episode_length": eval_percentile(lengths, 0.5),
    }


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: TrainConfig,
    valid_cells_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = observations.shape[0]
    indices = torch.arange(batch_size, device=observations.device)
    loss_tensors = {"policy": [], "value": [], "entropy": [], "approx_kl": []}
    steps = 0

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _epoch in range(config.update_epochs):
        permutation = indices[torch.randperm(batch_size, device=observations.device)]
        for start in range(0, batch_size, config.minibatch_size):
            mb = permutation[start:start + config.minibatch_size]
            logits, new_values = model(observations[mb])
            dist = masked_categorical(logits, observations[mb], valid_cells_mask)
            new_logprobs = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            logratio = new_logprobs - old_logprobs[mb]
            ratio = logratio.exp()
            pg_loss_unclipped = -advantages[mb] * ratio
            pg_loss_clipped = -advantages[mb] * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
            policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()
            value_loss = 0.5 * (new_values - returns[mb]).pow(2).mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean()
            loss_tensors["policy"].append(policy_loss.detach())
            loss_tensors["value"].append(value_loss.detach())
            loss_tensors["entropy"].append(entropy.detach())
            loss_tensors["approx_kl"].append(approx_kl.detach())
            steps += 1

    if steps == 0:
        return {key: torch.zeros((), device=observations.device) for key in loss_tensors}
    return {
        key: torch.stack(values).mean()
        for key, values in loss_tensors.items()
    }


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    scheduler = CurriculumScheduler(unlock_interval=config.unlock_interval)

    sync_gpu = bool(config.sync_gpu_step)
    active_size = scheduler.get_size(0)
    vec = create_vec(active_size, config, sync_gpu)
    validate_env_shape(vec)
    device = vec.device
    valid_cells_mask = compute_cells_mask(BOARD_SIZE, active_size, active_size).to(device)

    checkpoint_model = ChainReactionNet(board_size=BOARD_SIZE).to(device)
    optimizer = torch.optim.AdamW(checkpoint_model.parameters(), lr=config.learning_rate)
    transfer_meta = load_init_checkpoint(checkpoint_model, optimizer, config)
    model = maybe_compile_model(checkpoint_model, config)

    run_id = str(int(1000 * time.time()))
    wandb_run = init_wandb(config, run_id)
    logs: list[dict[str, float | int | str | dict]] = []
    global_step = 0
    update = 0
    start_time = time.time()
    interval = IntervalAverager()

    print(
        " ".join(
            [
                f"run_id={run_id}",
                f"board_size={BOARD_SIZE}",
                f"active={active_size}x{active_size}",
                f"total_agents={config.total_agents}",
                f"horizon={config.horizon}",
                f"minibatch_size={config.minibatch_size}",
                f"total_timesteps={config.total_timesteps}",
                f"unlock_interval={config.unlock_interval}",
                f"eval_interval={config.eval_interval}",
                f"sweep_interval={config.sweep_interval}",
                f"compile_model={config.compile_model}",
                f"device={device}",
            ]
        ),
        flush=True,
    )

    obs_buf = torch.zeros(config.horizon, vec.total_agents, vec.obs_size, device=device)
    action_buf = torch.zeros(config.horizon, vec.total_agents, dtype=torch.long, device=device)
    logprob_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
    value_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
    reward_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
    terminal_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
    illegal_mask = torch.zeros(config.horizon, vec.total_agents, dtype=torch.bool, device=device)

    try:
        while global_step < config.total_timesteps:
            # --- size selection and env switching ---
            new_size = scheduler.get_size(update)
            new_unlocked = scheduler.newly_unlocked(update)
            for s in new_unlocked:
                print(f"unlocked {s}x{s} at update {update}", flush=True)

            if new_size != active_size:
                prev = active_size
                vec.close()
                vec = create_vec(new_size, config, sync_gpu)
                validate_env_shape(vec)
                device = vec.device
                valid_cells_mask = compute_cells_mask(BOARD_SIZE, new_size, new_size).to(device)
                active_size = new_size

            # --- rollout ---
            rollout_start = time.perf_counter()
            for t in range(config.horizon):
                obs = vec.observations.float()
                with torch.no_grad():
                    actions, logprobs, values = evaluate_policy(model, obs, valid_cells_mask)
                obs_buf[t] = obs
                action_buf[t] = actions
                logprob_buf[t] = logprobs
                value_buf[t] = values

                illegal_mask[t] = (obs.gather(1, actions.unsqueeze(1)).squeeze(1) < 0) | (~valid_cells_mask[actions])
                vec.step(actions)
                reward_buf[t] = vec.rewards
                terminal_buf[t] = vec.terminals

            illegal_selected = illegal_mask.sum()
            rollout_elapsed = time.perf_counter() - rollout_start

            with torch.no_grad():
                _, last_values = model(vec.observations.float())
                advantages, returns = compute_negamax_gae(
                    reward_buf, value_buf, terminal_buf, last_values,
                    config.gamma, config.gae_lambda,
                )

            # --- PPO update ---
            update_start = time.perf_counter()
            flat_obs = obs_buf.reshape(-1, vec.obs_size)
            flat_actions = action_buf.reshape(-1)
            flat_logprobs = logprob_buf.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            loss_logs = ppo_update(
                model, optimizer, flat_obs, flat_actions, flat_logprobs,
                flat_advantages, flat_returns, config, valid_cells_mask,
            )
            update_elapsed = time.perf_counter() - update_start

            update += 1
            global_step += config.horizon * vec.total_agents
            elapsed = time.time() - start_time
            interval.add({
                "SPS": global_step / max(elapsed, 1e-6),
                "illegal_action_rate": illegal_selected / (config.horizon * vec.total_agents),
                "rollout_ms": rollout_elapsed * 1000,
                "update_ms": update_elapsed * 1000,
                "loss/policy": loss_logs["policy"],
                "loss/value": loss_logs["value"],
                "loss/entropy": loss_logs["entropy"],
                "loss/approx_kl": loss_logs["approx_kl"],
            })

            should_log = update % config.log_interval == 0 or global_step >= config.total_timesteps
            if should_log:
                env_logs = vec.log()
                averaged = interval.pop()
                averaged.update(flatten_log({"env": env_logs}))
                averaged["update"] = update
                averaged["agent_steps"] = global_step
                averaged["uptime"] = elapsed
                averaged["SPS"] = global_step / max(elapsed, 1e-6)
                averaged["active_size"] = active_size
                print(
                    " ".join(
                        [
                            f"update={update}",
                            f"size={active_size}",
                            f"agent_steps={global_step}",
                            f"SPS={averaged['SPS']:.0f}",
                            f"rollout_ms={averaged.get('rollout_ms', 0):.0f}",
                            f"update_ms={averaged.get('update_ms', 0):.0f}",
                            f"illegal_action_rate={averaged.get('illegal_action_rate', 0.0)}",
                            f"policy_loss={averaged.get('loss/policy')}",
                            f"value_loss={averaged.get('loss/value')}",
                            f"entropy={averaged.get('loss/entropy')}",
                            f"approx_kl={averaged.get('loss/approx_kl')}",
                        ]
                    ),
                    flush=True,
                )

            # --- periodic eval: largest unlocked size ---
            if update % config.eval_interval == 0 or global_step >= config.total_timesteps:
                eval_size = scheduler.unlocked_sizes(update)[-1]
                with torch.no_grad():
                    result = run_eval(checkpoint_model, eval_size, config, update, config.eval_games)
                print(
                    " ".join(
                        [
                            f"eval_update={update}",
                            f"size={eval_size}",
                            f"games={result['games']}",
                            f"combined_winrate={result['combined_winrate']}",
                            f"wilson_lower_bound={result['wilson_lower_bound']}",
                            f"illegal_selected_actions={result['illegal_selected_actions']}",
                            f"truncations={result['truncations']}",
                            f"mean_episode_length={result['mean_episode_length']}",
                        ]
                    ),
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log({f"eval/{k}": v for k, v in result.items() if isinstance(v, (float, int))}, step=global_step)

            # --- sweep eval: all unlocked sizes ---
            if update % config.sweep_interval == 0:
                sweep_sizes = scheduler.unlocked_sizes(update)
                print(f"sweep_eval_update={update} sizes={sweep_sizes}", flush=True)
                for sz in sweep_sizes:
                    with torch.no_grad():
                        result = run_eval(checkpoint_model, sz, config, update, config.sweep_games)
                    print(
                        " ".join(
                            [
                                f"sweep_eval size={sz}",
                                f"games={result['games']}",
                                f"combined_winrate={result['combined_winrate']}",
                                f"wilson_lower_bound={result['wilson_lower_bound']}",
                                f"truncations={result['truncations']}",
                            ]
                        ),
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log({f"sweep/{sz}/{k}": v for k, v in result.items() if isinstance(v, (float, int))}, step=global_step)

            if wandb_run is not None and should_log:
                wandb_run.log(averaged, step=global_step)

            # --- checkpoint ---
            if update % config.checkpoint_interval == 0 or global_step >= config.total_timesteps:
                path = os.path.join(config.checkpoint_dir, f"{run_id}_{global_step:016d}.pt")
                checkpoint_payload = {
                    "model": checkpoint_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "step": global_step,
                    "board_size": BOARD_SIZE,
                    "active_width": active_size,
                    "active_height": active_size,
                }
                if transfer_meta is not None:
                    checkpoint_payload.update({
                        "transfer_source_checkpoint": transfer_meta["source_checkpoint"],
                        "transfer_source_active_width": transfer_meta["source_active_width"],
                        "transfer_source_active_height": transfer_meta["source_active_height"],
                    })
                torch.save(checkpoint_payload, path)

        # --- final sweep eval ---
        print("final sweep eval", flush=True)
        all_sizes = CurriculumScheduler.SIZES
        for sz in all_sizes:
            with torch.no_grad():
                result = run_eval(checkpoint_model, sz, config, update, config.sweep_games)
            print(
                " ".join(
                    [
                        f"final_eval size={sz}",
                        f"games={result['games']}",
                        f"combined_winrate={result['combined_winrate']}",
                        f"wilson_lower_bound={result['wilson_lower_bound']}",
                        f"truncations={result['truncations']}",
                    ]
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({f"final/{sz}/{k}": v for k, v in result.items() if isinstance(v, (float, int))}, step=global_step)

        with open(os.path.join(config.log_dir, run_id + ".json"), "w") as f:
            json.dump({"config": asdict(config), "logs": logs}, f)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        vec.close()


if __name__ == "__main__":
    main()
