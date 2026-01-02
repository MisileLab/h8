from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .block import RetNetBlock
from .config import ModelConfig
from .decay_policy import build_decay_policy
from .norms import RMSNorm


class RetNetLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = []
        for idx in range(config.n_layers):
            decay_policy = build_decay_policy(
                config.decay, config.n_heads, config.d_model
            )
            block = RetNetBlock(
                config.d_model,
                config.retention_config(),
                config.ffn_mult,
                decay_policy,
                idx,
                config.norm_eps,
            )
            self.blocks.append(block)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        tokens: Any,
        *,
        mode: str = "parallel",
        chunk_size: int | None = None,
    ) -> Any:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, mode=mode, chunk_size=chunk_size)
        x = self.final_norm(x)
        return self.lm_head(x)

    def decay_reg_loss(self) -> Any:
        reg = mx.array(0.0, dtype=mx.float32)
        for block in self.blocks:
            reg = reg + block.decay_reg_loss()
        return reg

    def forward_step(
        self, token: Any, cache: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if token.ndim == 1:
            token = token[:, None]
        x = self.embed(token)

        if cache is None or "past_kv" not in cache:
            past_list = [None for _ in range(len(self.blocks))]
        else:
            past_list = cache["past_kv"]

        new_past = []
        for block, past_kv in zip(self.blocks, past_list):
            x, past_kv = block.forward_step(x, past_kv)
            new_past.append(past_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        cache = {"past_kv": new_past}
        return logits[:, 0], cache
