from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossMixConfig:
    alpha: float
    beta: float
    gamma: float
    warmup_frac: float


def warmup_scale(step: int, total_steps: int, warmup_frac: float) -> float:
    if warmup_frac <= 0:
        return 1.0
    warmup_steps = max(1, int(total_steps * warmup_frac))
    return min(1.0, step / warmup_steps)


def mix_losses(
    distill: torch.Tensor,
    rank: torch.Tensor,
    struct: torch.Tensor,
    config: LossMixConfig,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    scale = warmup_scale(step, total_steps, config.warmup_frac)
    return config.alpha * distill + scale * (config.beta * rank + config.gamma * struct)
