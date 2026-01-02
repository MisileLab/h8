from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecayConfig:
    policy: str = "learned_static"
    gamma_min: float = 0.1
    gamma_max: float = 0.999
    monotonic_heads: bool = True
    h_min: float = 1.0
    min_head_gap: float = 0.1
    decay_reg_strength: float = 0.0
    conditional_hidden_dim: int = 128
    eps: float = 1e-6


@dataclass
class RetentionConfig:
    n_heads: int = 8
    d_qk: int = 64
    d_v: int = 128
    decay: DecayConfig = field(default_factory=DecayConfig)
    norm_eps: float = 1e-5
    gate: str = "sigmoid"


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_qk: int = 64
    d_v: int = 128
    ffn_mult: int = 4
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.0
    decay: DecayConfig = field(default_factory=DecayConfig)

    def retention_config(self) -> RetentionConfig:
        return RetentionConfig(
            n_heads=self.n_heads,
            d_qk=self.d_qk,
            d_v=self.d_v,
            decay=self.decay,
            norm_eps=self.norm_eps,
        )
