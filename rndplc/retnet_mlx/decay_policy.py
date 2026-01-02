from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import ops
from .config import DecayConfig


class DecayPolicy(nn.Module):
    def __init__(self, config: DecayConfig, n_heads: int, d_model: int) -> None:
        super().__init__()
        self.config = config
        self.n_heads = n_heads
        self.d_model = d_model

    def alpha(self, x_normed: Any, mode: str, *, layer_idx: int) -> Any:
        raise NotImplementedError

    def reg_loss(self) -> Any:
        return mx.array(0.0, dtype=mx.float32)

    def chunk_products(self, alpha_chunk: Any) -> tuple[Any, Any]:
        alpha_chunk = ops.ensure_fp32(alpha_chunk)
        inner_decay = mx.cumprod(alpha_chunk, axis=-1)
        chunk_decay = mx.prod(alpha_chunk, axis=-1)
        return chunk_decay, inner_decay


class FixedDecayPolicy(DecayPolicy):
    def __init__(self, config: DecayConfig, n_heads: int, d_model: int) -> None:
        super().__init__(config, n_heads, d_model)
        self._gamma_values = self._build_gamma_values()

    def _build_gamma_values(self) -> list[float]:
        gamma_min = self.config.gamma_min
        gamma_max = self.config.gamma_max
        if gamma_min <= 0.0 or gamma_max >= 1.0:
            raise ValueError("gamma_min must be > 0 and gamma_max must be < 1")
        if self.n_heads == 1:
            return [gamma_max]
        log_min = math.log(gamma_min)
        log_max = math.log(gamma_max)
        values = []
        for idx in range(self.n_heads):
            ratio = idx / (self.n_heads - 1)
            values.append(math.exp(log_min + (log_max - log_min) * ratio))
        return values

    def gamma(self) -> Any:
        return mx.array(self._gamma_values, dtype=mx.float32)

    def alpha(self, x_normed: Any, mode: str, *, layer_idx: int) -> Any:
        return self.gamma()


class LearnedStaticDecayPolicy(DecayPolicy):
    def __init__(self, config: DecayConfig, n_heads: int, d_model: int) -> None:
        super().__init__(config, n_heads, d_model)
        self.u = mx.zeros((n_heads,), dtype=mx.float32)

    def _half_life(self) -> Any:
        inc = nn.softplus(self.u) + self.config.eps
        return self.config.h_min + mx.cumsum(inc)

    def gamma(self) -> Any:
        gamma_min = self.config.gamma_min
        gamma_max = self.config.gamma_max
        if gamma_min <= 0.0 or gamma_max >= 1.0:
            raise ValueError("gamma_min must be > 0 and gamma_max must be < 1")

        if self.config.monotonic_heads:
            half_life = self._half_life()
            gamma = mx.exp(-1.0 / half_life)
        else:
            gamma = mx.sigmoid(self.u)

        gamma = gamma_min + (gamma_max - gamma_min) * gamma
        return mx.clip(gamma, gamma_min, gamma_max)

    def alpha(self, x_normed: Any, mode: str, *, layer_idx: int) -> Any:
        return self.gamma()

    def reg_loss(self) -> Any:
        if self.config.decay_reg_strength <= 0.0:
            return mx.array(0.0, dtype=mx.float32)
        if not self.config.monotonic_heads:
            return mx.array(0.0, dtype=mx.float32)

        half_life = self._half_life()
        gaps = half_life[1:] - half_life[:-1]
        penalty = mx.maximum(self.config.min_head_gap - gaps, 0.0)
        return self.config.decay_reg_strength * mx.mean(penalty)


class ConditionalDecayPolicy(DecayPolicy):
    def __init__(self, config: DecayConfig, n_heads: int, d_model: int) -> None:
        super().__init__(config, n_heads, d_model)
        hidden = config.conditional_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_heads),
        )

    def alpha(self, x_normed: Any, mode: str, *, layer_idx: int) -> Any:
        x_normed = ops.ensure_fp32(x_normed)
        logits = self.mlp(x_normed)
        gamma_min = self.config.gamma_min
        gamma_max = self.config.gamma_max
        alpha = gamma_min + (gamma_max - gamma_min) * mx.sigmoid(logits)
        return mx.transpose(alpha, (0, 2, 1))


def build_decay_policy(config: DecayConfig, n_heads: int, d_model: int) -> DecayPolicy:
    if config.policy == "fixed":
        return FixedDecayPolicy(config, n_heads, d_model)
    if config.policy == "learned_static":
        return LearnedStaticDecayPolicy(config, n_heads, d_model)
    if config.policy == "conditional":
        return ConditionalDecayPolicy(config, n_heads, d_model)
    raise ValueError(f"Unknown decay policy: {config.policy}")
