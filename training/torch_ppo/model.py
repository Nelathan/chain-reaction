from __future__ import annotations

import torch
from torch import nn


DEFAULT_BOARD_SIZE = 8
INPUT_CHANNELS = 4


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(x)))
        return x + y


class ChainReactionNet(nn.Module):
    def __init__(
        self,
        board_size: int = DEFAULT_BOARD_SIZE,
        channels: int = 32,
        blocks: int = 3,
        value_channels: int = 8,
        value_hidden: int = 64,
    ):
        super().__init__()
        if board_size <= 0:
            raise ValueError(f"board_size must be positive, got {board_size}")
        self.board_size = board_size
        self.action_count = board_size * board_size
        self.register_buffer("critical_mass", self._critical_mass_map(board_size), persistent=False)
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(*[
            ResidualBlock(channels=channels)
            for _ in range(blocks)
        ])

        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)

        self.value_projection = nn.Conv2d(channels, value_channels, kernel_size=1)
        self.value_act = nn.SiLU()
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(value_channels, value_hidden),
            nn.SiLU(),
            nn.Linear(value_hidden, 1),
        )

    @staticmethod
    def _critical_mass_map(board_size: int) -> torch.Tensor:
        mass = torch.empty(1, 1, board_size, board_size)
        for y in range(board_size):
            for x in range(board_size):
                neighbors = 4
                if x == 0:
                    neighbors -= 1
                if x == board_size - 1:
                    neighbors -= 1
                if y == 0:
                    neighbors -= 1
                if y == board_size - 1:
                    neighbors -= 1
                mass[0, 0, y, x] = float(neighbors)
        return mass

    def _board(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2:
            observations = observations.view(observations.shape[0], 1, self.board_size, self.board_size)
        elif observations.ndim == 3:
            observations = observations.unsqueeze(1)
        if observations.shape[-2:] != (self.board_size, self.board_size):
            raise ValueError(f"expected {self.board_size}x{self.board_size} observations, got {tuple(observations.shape)}")
        return observations.float()

    def _input(self, observations: torch.Tensor) -> torch.Tensor:
        board = self._board(observations)
        capacity = self.critical_mass.to(dtype=board.dtype).expand(board.shape[0], -1, -1, -1)
        occupied = board != 0
        token_count = torch.where(occupied, capacity - board.abs(), torch.zeros_like(board))
        own_count = torch.where(board > 0, token_count, torch.zeros_like(board)) / 4.0
        opponent_count = torch.where(board < 0, token_count, torch.zeros_like(board)) / 4.0
        closeness = torch.where(occupied, (4.0 - board.abs()) / 3.0, torch.zeros_like(board))
        signed_closeness = closeness * board.sign()
        return torch.cat((capacity / 4.0, own_count, opponent_count, signed_closeness), dim=1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._input(observations).to(dtype=next(self.parameters()).dtype)
        x = self.trunk(self.stem(x))

        policy = self.policy_head(x)
        logits = policy.flatten(start_dim=1)

        value_features = self.value_act(self.value_projection(x))
        values = self.value_head(self.value_pool(value_features)).squeeze(-1)
        return logits, values
