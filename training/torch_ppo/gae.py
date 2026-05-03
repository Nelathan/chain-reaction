from __future__ import annotations

import torch


def compute_negamax_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminals: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE for alternating current-player-perspective turns.

    `values[t]` estimates the position for the player to move at timestep t.
    Rewards are also from the actor's timestep-t perspective. On a nonterminal
    transition, timestep t+1 is from the opponent's perspective, so the return
    recurrence is `G_t = r_t - gamma * G_{t+1}` rather than the usual
    single-agent `r_t + gamma * G_{t+1}`. The next value and next advantage
    therefore enter with a negamax sign flip.
    """
    if rewards.shape != values.shape or rewards.shape != terminals.shape:
        raise ValueError("rewards, values, and terminals must have matching [T, B] shapes")
    if last_value.shape != rewards.shape[1:]:
        raise ValueError("last_value must have shape [B]")

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(last_value)
    next_value = last_value

    for t in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - terminals[t]
        delta = rewards[t] - gamma * next_value * nonterminal - values[t]
        advantages[t] = delta - gamma * gae_lambda * nonterminal * next_advantage
        next_value = values[t]
        next_advantage = advantages[t]

    returns = advantages + values
    return advantages, returns
