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


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: TrainConfig,
) -> dict[str, float]:
    batch_size = observations.shape[0]
    indices = torch.arange(batch_size, device=observations.device)
    losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "approx_kl": 0.0}
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
            losses["policy"] += float(policy_loss.detach())
            losses["value"] += float(value_loss.detach())
            losses["entropy"] += float(entropy.detach())
            losses["approx_kl"] += float(approx_kl.detach())
            steps += 1

    return {key: value / max(steps, 1) for key, value in losses.items()}


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    vec = PufferVec(load_puffer_args(config))
    device = vec.device
    model = ChainReactionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    run_id = str(int(1000 * time.time()))
    logs: list[dict[str, float | int | str | dict]] = []
    global_step = 0
    update = 0
    start_time = time.time()

    try:
        while global_step < config.total_timesteps:
            obs_buf = torch.zeros(config.horizon, vec.total_agents, vec.obs_size, device=device)
            action_buf = torch.zeros(config.horizon, vec.total_agents, dtype=torch.long, device=device)
            logprob_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
            value_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
            reward_buf = torch.zeros(config.horizon, vec.total_agents, device=device)
            terminal_buf = torch.zeros(config.horizon, vec.total_agents, device=device)

            illegal_selected = 0
            for t in range(config.horizon):
                obs = torch.as_tensor(vec.observations, device=device).float()
                with torch.no_grad():
                    actions, logprobs, values = evaluate_policy(model, obs)
                obs_buf[t] = obs
                action_buf[t] = actions
                logprob_buf[t] = logprobs
                value_buf[t] = values

                illegal_selected += int((obs.gather(1, actions.view(-1, 1)).squeeze(1) < 0).sum().item())
                vec.step(actions)
                reward_buf[t] = torch.as_tensor(vec.rewards, device=device).float()
                terminal_buf[t] = torch.as_tensor(vec.terminals, device=device).float()

            with torch.no_grad():
                _, last_values = model(torch.as_tensor(vec.observations, device=device).float())
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
            env_logs = vec.log()
            log = {
                "run_id": run_id,
                "update": update,
                "agent_steps": global_step,
                "uptime": elapsed,
                "SPS": global_step / max(elapsed, 1e-6),
                "illegal_action_rate": illegal_selected / (config.horizon * vec.total_agents),
                "loss": loss_logs,
                "env": env_logs,
            }
            logs.append(log)
            print(json.dumps(log, sort_keys=True), flush=True)

            if update % config.checkpoint_interval == 0 or global_step >= config.total_timesteps:
                path = os.path.join(config.checkpoint_dir, f"{run_id}_{global_step:016d}.pt")
                torch.save({"model": model.state_dict(), "config": asdict(config), "step": global_step}, path)

        with open(os.path.join(config.log_dir, run_id + ".json"), "w") as f:
            json.dump({"config": asdict(config), "logs": logs}, f)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
