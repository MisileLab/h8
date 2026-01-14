from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor, p=2, dim=-1)


class LinearProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, normalize: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.normalize = normalize

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        if self.normalize:
            outputs = _l2_normalize(outputs)
        return outputs


class MLPProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.normalize = normalize

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.network(inputs)
        if self.normalize:
            outputs = _l2_normalize(outputs)
        return outputs


@dataclass
class StudentConfig:
    model_type: str
    in_dim: int
    out_dim: int
    hidden_dim: Optional[int]
    dropout: float
    normalize: bool


def build_student(config: StudentConfig) -> nn.Module:
    if config.model_type == "linear":
        return LinearProjector(
            config.in_dim, config.out_dim, normalize=config.normalize
        )
    if config.model_type == "mlp":
        if config.hidden_dim is None:
            raise ValueError("hidden_dim required for MLP")
        return MLPProjector(
            config.in_dim,
            config.hidden_dim,
            config.out_dim,
            dropout=config.dropout,
            normalize=config.normalize,
        )
    raise ValueError(f"Unknown model_type {config.model_type}")
