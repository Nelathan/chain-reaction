from __future__ import annotations

import torch
from torch import nn


BOARD_SIZE = 8
ACTION_COUNT = BOARD_SIZE * BOARD_SIZE


class PreActResidualBlock(nn.Module):
    def __init__(self, channels: int = 32, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act1(self.norm1(x)))
        y = self.conv2(self.act2(self.norm2(y)))
        return x + y


class ChainReactionNet(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        blocks: int = 4,
        groups: int = 8,
        value_channels: int = 8,
        value_hidden: int = 64,
    ):
        super().__init__()
        self.stem = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(*[
            PreActResidualBlock(channels=channels, groups=groups)
            for _ in range(blocks)
        ])

        self.policy_norm = nn.GroupNorm(groups, channels)
        self.policy_act = nn.SiLU()
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)

        self.value_norm = nn.GroupNorm(groups, channels)
        self.value_act = nn.SiLU()
        self.value_projection = nn.Conv2d(channels, value_channels, kernel_size=1)
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(value_channels, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, 1),
        )

    def _board(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2:
            observations = observations.view(observations.shape[0], 1, BOARD_SIZE, BOARD_SIZE)
        elif observations.ndim == 3:
            observations = observations.unsqueeze(1)
        if observations.shape[-2:] != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f"expected 8x8 observations, got {tuple(observations.shape)}")
        return observations.float()

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._board(observations)
        x = self.trunk(self.stem(x))

        policy = self.policy_head(self.policy_act(self.policy_norm(x)))
        logits = policy.flatten(start_dim=1)

        value_features = self.value_projection(self.value_act(self.value_norm(x)))
        values = self.value_head(self.value_pool(value_features)).squeeze(-1)
        return logits, values
