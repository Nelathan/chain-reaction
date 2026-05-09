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

from flashoptim import FlashAdamW, cast_model

from training.torch_ppo.gae import compute_negamax_gae
from training.torch_ppo.masking import apply_mask_to_logits, compute_cells_mask, legal_action_mask
from training.torch_ppo.model import ChainReactionNet
from training.torch_ppo.puffer_vec import PufferVec
from training.torch_ppo.evaluate_checkpoint import (
    percentile as eval_percentile,
    run_seat as eval_run_seat,
    split_games as eval_split_games,
    wilson_lower_bound as eval_wilson_lower_bound,
)


DEFAULT_BOARD_SIZE = 8
COMPILE_MIN_UPDATES = 100

# per-size max_turns caps from random-vs-random probes
SIZE_CAPS = {3: 16, 4: 32, 5: 64, 6: 80, 7: 104, 8: 136}


@dataclass
class TrainConfig:
    board_size: int = DEFAULT_BOARD_SIZE
    total_timesteps: int = 32_000_000
    total_agents: int = 1024
    horizon: int = 32
    minibatch_size: int = 32768
    update_epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
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
    compile_model: int = -1
    compile_mode: str = "default"
    sync_gpu_step: int = 0
    sync_timing: int = 0


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


def load_puffer_args(config: TrainConfig) -> dict:
    import pufferlib.pufferl

    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        args = pufferlib.pufferl.load_config("chain_reaction")
    finally:
        sys.argv = argv
    args["vec"]["total_agents"] = config.total_agents
    args["vec"]["num_buffers"] = 1
    args["env"]["max_turns"] = SIZE_CAPS.get(config.board_size, 128)
    args["env"]["active_width"] = config.board_size
    args["env"]["active_height"] = config.board_size
    args["train"]["horizon"] = config.horizon
    args["train"]["minibatch_size"] = config.minibatch_size
    args["train"]["total_timesteps"] = config.total_timesteps
    return args


def create_vec(config: TrainConfig, sync_gpu: bool) -> PufferVec:
    return PufferVec(load_puffer_args(config), sync_gpu_step=sync_gpu)


def evaluate_policy(model: nn.Module, obs: torch.Tensor, valid_cells_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = model(obs)
    legal_mask = legal_action_mask(obs, valid_cells_mask)
    masked_logits = apply_mask_to_logits(logits, legal_mask)
    probs = torch.softmax(masked_logits, dim=1)
    actions = torch.multinomial(probs, 1).squeeze(1)
    selected_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    logprobs = selected_probs.clamp_min(torch.finfo(selected_probs.dtype).tiny).log()
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
        self.tensor_totals: dict[str, torch.Tensor] = {}
        self.tensor_counts: dict[str, int] = {}

    def add(self, values: dict[str, float | int | torch.Tensor]) -> None:
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                detached = value.detach()
                if key in self.tensor_totals:
                    self.tensor_totals[key] = self.tensor_totals[key] + detached
                    self.tensor_counts[key] += 1
                else:
                    self.tensor_totals[key] = detached
                    self.tensor_counts[key] = 1
            elif isinstance(value, (float, int)):
                self.totals[key] = self.totals.get(key, 0.0) + float(value)
                self.counts[key] = self.counts.get(key, 0) + 1

    def pop(self) -> dict[str, float]:
        averaged = {
            key: self.totals[key] / max(self.counts.get(key, 0), 1)
            for key in self.totals
        }
        for key, value in self.tensor_totals.items():
            averaged[key] = float((value / max(self.tensor_counts.get(key, 0), 1)).item())
        self.totals.clear()
        self.counts.clear()
        self.tensor_totals.clear()
        self.tensor_counts.clear()
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


def should_compile_model(config: TrainConfig) -> bool:
    if config.compile_model < 0:
        planned_updates = max(config.total_timesteps // max(config.total_agents * config.horizon, 1), 1)
        return planned_updates >= COMPILE_MIN_UPDATES
    return config.compile_model != 0


def maybe_compile_model(model: nn.Module, config: TrainConfig) -> nn.Module:
    if not should_compile_model(config):
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("CHAIN_REACTION_COMPILE_MODEL=1 requires torch.compile")
    return torch.compile(model, mode=config.compile_mode)


def sync_for_timing(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize()


def validate_env_shape(vec: PufferVec, board_size: int) -> None:
    expected_cells = board_size * board_size
    if vec.obs_size != expected_cells:
        raise RuntimeError(
            f"env obs_size {vec.obs_size} does not match board_size "
            f"{board_size} ({expected_cells} cells); rebuild the Ocean env "
            "with matching CR_WIDTH/CR_HEIGHT"
        )


def load_init_checkpoint(model: nn.Module, optimizer: FlashAdamW, config: TrainConfig) -> dict[str, int | str | None] | None:
    if not config.init_checkpoint:
        return None

    checkpoint_path = config.init_checkpoint
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"init checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Restore FP32 weights into the bf16 model with error-bit recomputation.
    optimizer.set_fp32_model_state_dict(model, checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint_config = checkpoint.get("config", {})
    return {
        "source_checkpoint": checkpoint_path,
        "source_board_size": checkpoint_config.get("board_size", config.board_size),
    }


def run_eval(
    model: nn.Module,
    board_size: int,
    config: TrainConfig,
    update: int,
    n_games: int,
) -> dict[str, float | int | None]:
    eval_agents = min(config.total_agents, 256)
    max_turns = SIZE_CAPS.get(board_size, 128)
    mask = compute_cells_mask(board_size, board_size, board_size)
    eval_args = Namespace(
        total_agents=eval_agents,
        max_turns=max_turns,
        seed=config.seed + update * 1000003,
        board_size=board_size,
        active_width=board_size,
        active_height=board_size,
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
    optimizer: FlashAdamW,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: TrainConfig,
    valid_cells_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = observations.shape[0]
    loss_tensors = {
        "policy": torch.zeros((), device=observations.device),
        "value": torch.zeros((), device=observations.device),
        "entropy": torch.zeros((), device=observations.device),
        "approx_kl": torch.zeros((), device=observations.device),
    }
    steps = 0

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _epoch in range(config.update_epochs):
        permutation = torch.randperm(batch_size, device=observations.device)
        shuffled_observations = observations[permutation]
        shuffled_actions = actions[permutation]
        shuffled_old_logprobs = old_logprobs[permutation]
        shuffled_advantages = advantages[permutation]
        shuffled_returns = returns[permutation]
        for start in range(0, batch_size, config.minibatch_size):
            stop = start + config.minibatch_size
            mb_observations = shuffled_observations[start:stop]
            mb_actions = shuffled_actions[start:stop]
            mb_old_logprobs = shuffled_old_logprobs[start:stop]
            mb_advantages = shuffled_advantages[start:stop]
            mb_returns = shuffled_returns[start:stop]

            logits, new_values = model(mb_observations)
            mb_legal_masks = legal_action_mask(mb_observations, valid_cells_mask)
            masked_logits = apply_mask_to_logits(logits, mb_legal_masks)
            log_probs = torch.log_softmax(masked_logits, dim=1)
            new_logprobs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()

            logratio = new_logprobs - mb_old_logprobs
            ratio = logratio.exp()
            pg_loss_unclipped = -mb_advantages * ratio
            pg_loss_clipped = -mb_advantages * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
            policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()
            value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, foreach=True)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean()
            loss_tensors["policy"] += policy_loss.detach()
            loss_tensors["value"] += value_loss.detach()
            loss_tensors["entropy"] += entropy.detach()
            loss_tensors["approx_kl"] += approx_kl.detach()
            steps += 1

    if steps == 0:
        return {key: torch.zeros((), device=observations.device) for key in loss_tensors}
    return {key: value / steps for key, value in loss_tensors.items()}


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sync_gpu = bool(config.sync_gpu_step)
    sync_timing = bool(config.sync_timing)
    compile_enabled = should_compile_model(config)
    vec = create_vec(config, sync_gpu)
    validate_env_shape(vec, config.board_size)
    device = vec.device
    valid_cells_mask = compute_cells_mask(config.board_size, config.board_size, config.board_size).to(device)

    checkpoint_model = ChainReactionNet(board_size=config.board_size).to(device)
    cast_model(checkpoint_model, dtype=torch.bfloat16)
    optimizer = FlashAdamW(
        checkpoint_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
                f"board_size={config.board_size}",
                f"total_agents={config.total_agents}",
                f"horizon={config.horizon}",
                f"minibatch_size={config.minibatch_size}",
                f"total_timesteps={config.total_timesteps}",
                f"eval_interval={config.eval_interval}",
                f"compile_enabled={compile_enabled}",
                f"sync_timing={sync_timing}",
                f"optimizer=FlashAdamW",
                f"device={device}",
            ]
        ),
        flush=True,
    )

    obs_buf = torch.empty(
        config.horizon, vec.total_agents, vec.obs_size,
        dtype=vec.observations.dtype,
        device=device,
    )
    action_buf = torch.empty(config.horizon, vec.total_agents, dtype=torch.long, device=device)
    logprob_buf = torch.empty(config.horizon, vec.total_agents, device=device)
    value_buf = torch.empty(config.horizon, vec.total_agents, device=device)
    reward_buf = torch.empty(config.horizon, vec.total_agents, device=device)
    terminal_buf = torch.empty(config.horizon, vec.total_agents, device=device)

    try:
        while global_step < config.total_timesteps:
            # --- rollout ---
            sync_for_timing(device, sync_timing)
            rollout_start = time.perf_counter()
            for t in range(config.horizon):
                obs = vec.observations
                with torch.no_grad():
                    actions, logprobs, values = evaluate_policy(model, obs, valid_cells_mask)
                obs_buf[t] = obs
                action_buf[t] = actions
                logprob_buf[t] = logprobs
                value_buf[t] = values

                vec.step(actions)
                reward_buf[t] = vec.rewards
                terminal_buf[t] = vec.terminals

            illegal_selected = (obs_buf.gather(2, action_buf.unsqueeze(-1)).squeeze(-1) < 0).sum()
            sync_for_timing(device, sync_timing)
            rollout_elapsed = time.perf_counter() - rollout_start

            with torch.no_grad():
                _, last_values = model(vec.observations)
                advantages, returns = compute_negamax_gae(
                    reward_buf, value_buf, terminal_buf, last_values,
                    config.gamma, config.gae_lambda,
                )

            # --- PPO update ---
            sync_for_timing(device, sync_timing)
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
            sync_for_timing(device, sync_timing)
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
                print(
                    " ".join(
                        [
                            f"update={update}",
                            f"board_size={config.board_size}",
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
                logs.append(dict(averaged))

            # --- periodic eval ---
            if update % config.eval_interval == 0 or global_step >= config.total_timesteps:
                with torch.no_grad():
                    result = run_eval(
                        checkpoint_model,
                        config.board_size,
                        config,
                        update,
                        config.eval_games,
                    )
                print(
                    " ".join(
                        [
                            f"eval_update={update}",
                            f"board_size={config.board_size}",
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

            if wandb_run is not None and should_log:
                wandb_run.log(averaged, step=global_step)

            # --- checkpoint ---
            if update % config.checkpoint_interval == 0 or global_step >= config.total_timesteps:
                path = os.path.join(config.checkpoint_dir, f"{run_id}_{global_step:016d}.pt")
                checkpoint_payload = {
                    "model": optimizer.get_fp32_model_state_dict(checkpoint_model),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "step": global_step,
                    "board_size": config.board_size,
                    "compile_enabled": compile_enabled,
                }
                if transfer_meta is not None:
                    checkpoint_payload.update({
                        "transfer_source_checkpoint": transfer_meta["source_checkpoint"],
                        "transfer_source_board_size": transfer_meta["source_board_size"],
                    })
                torch.save(checkpoint_payload, path)

        with open(os.path.join(config.log_dir, run_id + ".json"), "w") as f:
            json.dump({"config": asdict(config), "logs": logs}, f)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        vec.close()


if __name__ == "__main__":
    main()
