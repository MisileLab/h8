from __future__ import annotations

from typing import Any

import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm: Any = nn.RMSNorm(dims, eps=eps)

    def __call__(self, x: Any) -> Any:
        return self.norm(x)


class TokenwiseGroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ) -> None:
        super().__init__()
        self.num_groups: int = num_groups
        self.dims: int = dims
        self.norm: Any = nn.GroupNorm(
            num_groups,
            dims,
            eps=eps,
            affine=affine,
            pytorch_compatible=pytorch_compatible,
        )

    def __call__(self, x: Any) -> Any:
        if x.ndim == 2:
            if x.shape[-1] != self.dims:
                raise ValueError(
                    f"TokenwiseGroupNorm expects last dim {self.dims}, got {x.shape[-1]}"
                )
            return self.norm(x)

        if x.ndim == 3:
            b, t, c = x.shape
            if c != self.dims:
                raise ValueError(
                    f"TokenwiseGroupNorm expects last dim {self.dims}, got {c}"
                )
            x_2d = x.reshape(b * t, c)
            y_2d = self.norm(x_2d)
            return y_2d.reshape(b, t, c)

        raise ValueError(f"TokenwiseGroupNorm expects rank 2 or 3, got {x.ndim}")
