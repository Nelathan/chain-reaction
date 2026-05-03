from __future__ import annotations

import torch


def flatten_observation(observations: torch.Tensor) -> torch.Tensor:
    if observations.ndim == 4:
        return observations.flatten(start_dim=1)
    if observations.ndim == 3:
        return observations.reshape(observations.shape[0], -1)
    if observations.ndim == 2:
        return observations
    raise ValueError(f"unsupported observation shape: {tuple(observations.shape)}")


def legal_action_mask(observations: torch.Tensor) -> torch.Tensor:
    """Return legal action mask from current-player-relative observations.

    Empty cells are 0 and current-player cells are positive. Opponent cells are
    negative, so legality is exactly obs >= 0.
    """
    return flatten_observation(observations) >= 0


def apply_legal_mask(logits: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    mask = legal_action_mask(observations).to(device=logits.device)
    if mask.shape != logits.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape)}")
    floor = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, floor)


def masked_categorical(logits: torch.Tensor, observations: torch.Tensor) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(logits=apply_legal_mask(logits, observations))
