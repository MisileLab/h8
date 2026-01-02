from __future__ import annotations

from typing import Any

import mlx.core as mx


def assert_rank(x: Any, rank: int, name: str = "tensor") -> None:
    if x.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {x.ndim}")


def assert_last_dim(x: Any, dim: int, name: str = "tensor") -> None:
    if x.shape[-1] != dim:
        raise ValueError(f"{name} last dim must be {dim}, got {x.shape[-1]}")


def assert_divisible(value: int, divisor: int, name: str = "value") -> None:
    if value % divisor != 0:
        raise ValueError(f"{name} must be divisible by {divisor}, got {value}")


def as_array(x: Any, dtype: Any | None = None) -> Any:
    if hasattr(x, "dtype"):
        if dtype is None or x.dtype == dtype:
            return x
        return x.astype(dtype)
    if dtype is None:
        return mx.array(x)
    return mx.array(x, dtype=dtype)


def cast_to(x: Any, dtype: Any) -> Any:
    return as_array(x, dtype)


def ensure_fp32(x: Any) -> Any:
    return cast_to(x, mx.float32)


def causal_mask(t: int, dtype: Any = mx.float32) -> Any:
    ones = mx.ones((t, t), dtype=dtype)
    return mx.tril(ones, k=0)


def _gamma_key(gamma: Any) -> object:
    if isinstance(gamma, (float, int)):
        return float(gamma)
    if hasattr(gamma, "tolist"):
        gamma = gamma.tolist()
    if isinstance(gamma, (list, tuple)):
        return tuple(float(x) for x in gamma)
    return float(gamma)


def build_decay_mask_from_gamma(gamma: Any, t: int, dtype: Any = mx.float32) -> Any:
    idx = mx.arange(t)
    i = idx[:, None]
    j = idx[None, :]
    power = (i - j).astype(dtype)
    mask = mx.where(i >= j, mx.ones((t, t), dtype=dtype), mx.zeros((t, t), dtype=dtype))

    if hasattr(gamma, "ndim") and getattr(gamma, "ndim") == 1:
        g = cast_to(gamma, dtype)[:, None, None]
        power = power[None, :, :]
        mask = mask[None, :, :]
        return mx.power(g, power) * mask

    g = mx.array(gamma, dtype=dtype)
    return mx.power(g, power) * mask


class DecayMaskCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[object, int, str], Any] = {}

    def get(self, gamma: Any, t: int, dtype: Any = mx.float32) -> Any:
        key = (_gamma_key(gamma), t, str(dtype))
        if key not in self._cache:
            self._cache[key] = build_decay_mask_from_gamma(gamma, t, dtype=dtype)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()
