from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from training.torch_ppo.gae import compute_negamax_gae
from training.torch_ppo.masking import masked_categorical
from training.torch_ppo.model import ChainReactionNet
from training.torch_ppo.puffer_vec import PufferVec


@dataclass
class TrainConfig:
    total_timesteps: int = 10_000_000
    total_agents: int = 4096
    horizon: int = 64
    minibatch_size: int = 32768
    update_epochs: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    max_turns: int = 4096
    seed: int = 73
    checkpoint_interval: int = 20
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
    log_interval: int = 10
    compile_model: int = 1
    compile_mode: str = "default"
    sync_gpu_step: int = 0


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
    args["env"]["max_turns"] = config.max_turns
    args["train"]["horizon"] = config.horizon
    args["train"]["minibatch_size"] = config.minibatch_size
    args["train"]["total_timesteps"] = config.total_timesteps
    return args


def evaluate_policy(model: nn.Module, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = model(obs)
    dist = masked_categorical(logits, obs)
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


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: TrainConfig,
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
            dist = masked_categorical(logits, observations[mb])
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

    vec = PufferVec(load_puffer_args(config), sync_gpu_step=bool(config.sync_gpu_step))
    device = vec.device
    checkpoint_model = ChainReactionNet().to(device)
    model = maybe_compile_model(checkpoint_model, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    run_id = str(int(1000 * time.time()))
    wandb_run = init_wandb(config, run_id)
    logs: list[dict[str, float | int | str | dict]] = []
    global_step = 0
    update = 0
    start_time = time.time()
    interval = IntervalAverager()

    obs_buf = torch.zeros(config.horizon, vec.total_agents, vec.obs_size, device=device)
    action_buf = torch.zeros(config.horizon, vec.total_agents, dtype=torch.long, device=device)
    logprob_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
    value_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
    reward_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
    terminal_buf = torch.zeros(config.horizon, vec.total_agents, device=device)

    try:
        while global_step < config.total_timesteps:
            illegal_selected = torch.zeros((), dtype=torch.float32, device=device)
            for t in range(config.horizon):
                obs = vec.observations.float()
                with torch.no_grad():
                    actions, logprobs, values = evaluate_policy(model, obs)
                obs_buf[t] = obs
                action_buf[t] = actions
                logprob_buf[t] = logprobs
                value_buf[t] = values

                illegal_selected += (obs.gather(1, actions.view(-1, 1)).squeeze(1) < 0).sum()
                vec.step(actions)
                reward_buf[t] = vec.rewards.float()
                terminal_buf[t] = vec.terminals.float()

            with torch.no_grad():
                _, last_values = model(vec.observations.float())
                advantages, returns = compute_negamax_gae(
                    reward_buf, value_buf, terminal_buf, last_values,
                    config.gamma, config.gae_lambda,
                )

            flat_obs = obs_buf.reshape(-1, vec.obs_size)
            flat_actions = action_buf.reshape(-1)
            flat_logprobs = logprob_buf.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            loss_logs = ppo_update(
                model, optimizer, flat_obs, flat_actions, flat_logprobs,
                flat_advantages, flat_returns, config,
            )

            update += 1
            global_step += config.horizon * vec.total_agents
            elapsed = time.time() - start_time
            interval.add({
                "SPS": global_step / max(elapsed, 1e-6),
                "illegal_action_rate": illegal_selected / (config.horizon * vec.total_agents),
                "loss/policy": loss_logs["policy"],
                "loss/value": loss_logs["value"],
                "loss/entropy": loss_logs["entropy"],
                "loss/approx_kl": loss_logs["approx_kl"],
            })

            log = {
                "run_id": run_id,
                "update": update,
                "agent_steps": global_step,
                "uptime": elapsed,
                "SPS": global_step / max(elapsed, 1e-6),
            }

            should_log = update % config.log_interval == 0 or global_step >= config.total_timesteps
            if should_log:
                env_logs = vec.log()
                averaged = interval.pop()
                averaged.update(flatten_log({"env": env_logs}))
                averaged["update"] = update
                averaged["agent_steps"] = global_step
                averaged["uptime"] = elapsed
                averaged["SPS"] = global_step / max(elapsed, 1e-6)
                log["env"] = env_logs
                log["interval"] = averaged

            logs.append(log)
            if wandb_run is not None and should_log:
                wandb_run.log(averaged, step=global_step)

            if update % config.checkpoint_interval == 0 or global_step >= config.total_timesteps:
                path = os.path.join(config.checkpoint_dir, f"{run_id}_{global_step:016d}.pt")
                torch.save({"model": checkpoint_model.state_dict(), "config": asdict(config), "step": global_step}, path)

        with open(os.path.join(config.log_dir, run_id + ".json"), "w") as f:
            json.dump({"config": asdict(config), "logs": logs}, f)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        vec.close()


if __name__ == "__main__":
    main()
