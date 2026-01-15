from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn.functional as F

import torch
from torch import nn


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor, p=2, dim=-1)


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


class TrackBProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 128,
        use_skip: bool = True,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_skip = use_skip

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()

        if use_skip:
            self.skip = nn.Linear(in_dim, out_dim, bias=False)
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.skip = None
            self.alpha = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.fc2(self.act(self.fc1(inputs)))
        if self.use_skip:
            if self.skip is None or self.alpha is None:
                raise RuntimeError("use_skip=True but skip/alpha not initialized")
            base = base + self.alpha * self.skip(inputs)
        return _l2_normalize(base)


@dataclass
class StudentConfig:
    model_type: str
    in_dim: int
    out_dim: int
    hidden_dim: Optional[int]
    dropout: float
    normalize: bool
    track_b_use_skip: bool = True
    track_b_alpha_init: float = 0.1


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
    if config.model_type == "track_b":
        return TrackBProjectionHead(
            in_dim=config.in_dim,
            hidden_dim=512,
            out_dim=128,
            use_skip=config.track_b_use_skip,
            alpha_init=config.track_b_alpha_init,
        )
    raise ValueError(f"Unknown model_type {config.model_type}")
