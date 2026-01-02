from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import RetentionConfig
from .decay_policy import DecayPolicy
from .norms import RMSNorm
from .retention import MultiHeadRetention

class RetNetBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        retention_config: RetentionConfig,
        ffn_mult: int,
        decay_policy: DecayPolicy,
        layer_idx: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.retention = MultiHeadRetention(
            d_model, retention_config, decay_policy, layer_idx
        )
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

        hidden = d_model * ffn_mult
        self.ffn_in = nn.Linear(d_model, hidden * 2)
        self.ffn_out = nn.Linear(hidden, d_model)

    def __call__(
        self,
        x: Any,
        *,
        mode: str = "parallel",
        chunk_size: int | None = None,
    ) -> Any:
        x_norm = self.norm1(x)
        ret_out, _ = self.retention(x_norm, mode=mode, chunk_size=chunk_size)
        x = x + ret_out

        x_norm = self.norm2(x)
        ffn = self.ffn_in(x_norm)
        gate, up = mx.split(ffn, 2, axis=-1)
        ffn = nn.silu(gate) * up
        ffn = self.ffn_out(ffn)
        x = x + ffn
        return x

    def decay_reg_loss(self) -> Any:
        return self.retention.decay_reg_loss()

    def forward_step(self, x: Any, past_kv: Any | None) -> tuple[Any, Any]:
        x_norm = self.norm1(x)
        ret_out, new_kv = self.retention.forward_step(x_norm, past_kv)
        x = x + ret_out

        x_norm = self.norm2(x)
        ffn = self.ffn_in(x_norm)
        gate, up = mx.split(ffn, 2, axis=-1)
        ffn = nn.silu(gate) * up
        ffn = self.ffn_out(ffn)
        x = x + ffn
        return x, new_kv
