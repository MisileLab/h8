from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from . import ops
from .config import RetentionConfig
from .decay_policy import DecayPolicy
from .norms import TokenwiseGroupNorm


def _broadcast_alpha(alpha: Any, batch: int, heads: int) -> Any:
    if alpha.ndim == 0:
        return mx.full((batch, heads), alpha, dtype=alpha.dtype)
    if alpha.ndim == 1:
        return mx.broadcast_to(alpha[None, :], (batch, heads))
    return alpha


def parallel_retention_fast_fixed(q: Any, k: Any, v: Any, decay_mask: Any) -> Any:
    q = ops.ensure_fp32(q)
    k = ops.ensure_fp32(k)
    v = ops.ensure_fp32(v)
    decay_mask = ops.ensure_fp32(decay_mask)

    if decay_mask.ndim == 2:
        decay = decay_mask[None, None, :, :]
    elif decay_mask.ndim == 3:
        decay = decay_mask[None, :, :, :]
    else:
        raise ValueError("decay_mask must have rank 2 or 3")

    qk = mx.einsum("bhtd,bhsd->bhts", q, k)
    weights = qk * decay
    return mx.einsum("bhts,bhsd->bhtd", weights, v)


def parallel_retention_general(q: Any, k: Any, v: Any, alpha_bht: Any) -> Any:
    q = ops.ensure_fp32(q)
    k = ops.ensure_fp32(k)
    v = ops.ensure_fp32(v)
    alpha_bht = ops.ensure_fp32(alpha_bht)

    b, h, t, _ = q.shape
    prefix = mx.cumprod(alpha_bht, axis=-1)
    prefix_i = prefix[:, :, :, None]
    prefix_j = prefix[:, :, None, :]
    decay = prefix_i / prefix_j

    causal = ops.causal_mask(t, dtype=alpha_bht.dtype)[None, None, :, :]
    decay = decay * causal

    qk = mx.einsum("bhtd,bhsd->bhts", q, k)
    weights = qk * decay
    return mx.einsum("bhts,bhsd->bhtd", weights, v)


def recurrent_retention_step(
    q_t: Any,
    k_t: Any,
    v_t: Any,
    past_kv: Any,
    alpha_t: Any,
) -> tuple[Any, Any]:
    q_t = ops.ensure_fp32(q_t)
    k_t = ops.ensure_fp32(k_t)
    v_t = ops.ensure_fp32(v_t)
    alpha_t = ops.ensure_fp32(alpha_t)

    b, h, _ = q_t.shape
    alpha_t = _broadcast_alpha(alpha_t, b, h)

    if past_kv is None:
        dv = v_t.shape[-1]
        past_kv = mx.zeros((b, h, q_t.shape[-1], dv), dtype=q_t.dtype)
    else:
        past_kv = ops.ensure_fp32(past_kv)

    kv_update = mx.einsum("bhd,bhe->bhde", k_t, v_t)
    new_kv = past_kv * alpha_t[:, :, None, None] + kv_update
    out_t = mx.einsum("bhd,bhde->bhe", q_t, new_kv)
    return out_t, new_kv


def chunkwise_retention(
    q: Any,
    k: Any,
    v: Any,
    past_kv: Any,
    alpha_chunk: Any,
) -> tuple[Any, Any]:
    q = ops.ensure_fp32(q)
    k = ops.ensure_fp32(k)
    v = ops.ensure_fp32(v)
    alpha_chunk = ops.ensure_fp32(alpha_chunk)

    b, h, c, _ = q.shape
    if past_kv is None:
        dv = v.shape[-1]
        past_kv = mx.zeros((b, h, q.shape[-1], dv), dtype=q.dtype)
    else:
        past_kv = ops.ensure_fp32(past_kv)

    inner_decay = mx.cumprod(alpha_chunk, axis=-1)
    chunk_decay = mx.prod(alpha_chunk, axis=-1)

    q_past = mx.einsum("bhcd,bhde->bhce", q, past_kv)
    cross = q_past * inner_decay[:, :, :, None]

    within = parallel_retention_general(q, k, v, alpha_chunk)
    out = cross + within

    weights = chunk_decay[:, :, None] / inner_decay
    v_weighted = v * weights[:, :, :, None]
    kv_update = mx.einsum("bhcd,bhce->bhde", k, v_weighted)
    new_kv = past_kv * chunk_decay[:, :, None, None] + kv_update
    return out, new_kv


class MultiHeadRetention(nn.Module):
    def __init__(
        self,
        d_model: int,
        config: RetentionConfig,
        decay_policy: DecayPolicy,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.decay_policy = decay_policy
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.d_qk = config.d_qk
        self.d_v = config.d_v
        self.decay_cache = ops.DecayMaskCache()

        self.q_proj = nn.Linear(d_model, config.n_heads * config.d_qk, bias=False)
        self.k_proj = nn.Linear(d_model, config.n_heads * config.d_qk, bias=False)
        self.v_proj = nn.Linear(d_model, config.n_heads * config.d_v, bias=False)
        self.gate_proj = nn.Linear(d_model, config.n_heads * config.d_v)
        self.out_proj = nn.Linear(config.n_heads * config.d_v, d_model)
        self.norm = TokenwiseGroupNorm(
            config.n_heads,
            config.n_heads * config.d_v,
            eps=config.norm_eps,
        )

    def _apply_gate(self, x: Any) -> Any:
        if self.config.gate == "sigmoid":
            return mx.sigmoid(x)
        if self.config.gate == "silu":
            return nn.silu(x)
        raise ValueError(f"Unknown gate activation: {self.config.gate}")

    def _project_qkv(self, x_normed: Any) -> tuple[Any, Any, Any]:
        b, t, _ = x_normed.shape
        q = self.q_proj(x_normed).reshape(b, t, self.n_heads, self.d_qk)
        k = self.k_proj(x_normed).reshape(b, t, self.n_heads, self.d_qk)
        v = self.v_proj(x_normed).reshape(b, t, self.n_heads, self.d_v)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        return q, k, v

    def _merge_heads(self, x: Any) -> Any:
        b, h, t, d = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(b, t, h * d)

    def __call__(
        self,
        x_normed: Any,
        *,
        mode: str = "parallel",
        chunk_size: int | None = None,
        past_kv: Any | None = None,
    ) -> tuple[Any, Any | None]:
        q, k, v = self._project_qkv(x_normed)
        b, h, t, _ = q.shape

        alpha = self.decay_policy.alpha(x_normed, mode, layer_idx=self.layer_idx)
        alpha = ops.ensure_fp32(alpha)

        if mode == "parallel":
            if alpha.ndim <= 1:
                decay_mask = self.decay_cache.get(alpha, t, dtype=mx.float32)
                out = parallel_retention_fast_fixed(q, k, v, decay_mask)
            else:
                out = parallel_retention_general(q, k, v, alpha)
            past_kv_out = None
        elif mode == "chunkwise":
            chunk = chunk_size or t
            outs = []
            past = past_kv
            if alpha.ndim <= 1:
                gamma = alpha
                alpha_full = None
            else:
                gamma = None
                alpha_full = alpha

            for start in range(0, t, chunk):
                end = min(start + chunk, t)
                q_chunk = q[:, :, start:end, :]
                k_chunk = k[:, :, start:end, :]
                v_chunk = v[:, :, start:end, :]
                if gamma is not None:
                    alpha_chunk = mx.broadcast_to(
                        gamma[None, :, None], (b, h, end - start)
                    )
                else:
                    alpha_chunk = alpha_full[:, :, start:end]

                out_chunk, past = chunkwise_retention(
                    q_chunk, k_chunk, v_chunk, past, alpha_chunk
                )
                outs.append(out_chunk)

            out = mx.concatenate(outs, axis=2)
            past_kv_out = past
        else:
            raise ValueError(f"Unknown retention mode: {mode}")

        out = self._merge_heads(out)
        out = self.norm(out)
        gate = self._apply_gate(self.gate_proj(x_normed))
        out = out * gate
        out = self.out_proj(out)
        return out, past_kv_out

    def decay_reg_loss(self) -> Any:
        return self.decay_policy.reg_loss()

    def forward_step(self, x_normed: Any, past_kv: Any | None) -> tuple[Any, Any]:
        q, k, v = self._project_qkv(x_normed)
        q_t = q[:, :, 0]
        k_t = k[:, :, 0]
        v_t = v[:, :, 0]

        alpha = self.decay_policy.alpha(x_normed, "recurrent", layer_idx=self.layer_idx)
        alpha = ops.ensure_fp32(alpha)
        if alpha.ndim == 3:
            alpha_t = alpha[:, :, 0]
        else:
            alpha_t = alpha

        out_t, new_kv = recurrent_retention_step(q_t, k_t, v_t, past_kv, alpha_t)
        out_t = out_t[:, :, None, :]
        out = self._merge_heads(out_t)
        out = self.norm(out)
        gate = self._apply_gate(self.gate_proj(x_normed))
        out = out * gate
        out = self.out_proj(out)
        return out, new_kv
