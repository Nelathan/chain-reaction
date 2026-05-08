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


def compute_cells_mask(board_size: int, active_width: int, active_height: int) -> torch.Tensor:
    """Return a 1D boolean tensor of length board_size² marking valid cells.

    Valid cells are the top-left active_width × active_height rectangle within
    the board_size × board_size grid. The mask is ANDed with the observation-sign
    legal mask so inactive cells are never sampled.
    """
    total = board_size * board_size
    mask = torch.zeros(total, dtype=torch.bool)
    for y in range(active_height):
        for x in range(active_width):
            mask[y * board_size + x] = True
    return mask


def legal_action_mask(
    observations: torch.Tensor,
    valid_cells_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return legal action mask from current-player-relative observations.

    Empty cells are 0 and current-player cells are positive. Opponent cells are
    negative, so legality is obs >= 0. If valid_cells_mask is provided, it is
    ANDed with the observation-sign mask so inactive cells are illegal.
    """
    flat = flatten_observation(observations)
    mask = flat >= 0
    if valid_cells_mask is not None:
        if valid_cells_mask.device != mask.device:
            valid_cells_mask = valid_cells_mask.to(device=mask.device)
        mask = mask & valid_cells_mask
    return mask


def apply_legal_mask(
    logits: torch.Tensor,
    observations: torch.Tensor,
    valid_cells_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = legal_action_mask(observations, valid_cells_mask).to(device=logits.device)
    return apply_mask_to_logits(logits, mask)


def apply_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != logits.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape)}")
    floor = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, floor)
