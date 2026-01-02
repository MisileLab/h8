from __future__ import annotations

from typing import Any

import mlx.nn as nn


class XPos(nn.Module):
    def __init__(self, kind: str = "none") -> None:
        super().__init__()
        self.kind = kind
        self._rope: nn.RoPE | None = None

    def apply_qk(self, q: Any, k: Any) -> tuple[Any, Any]:
        if self.kind == "none":
            return q, k
        if self.kind == "rope":
            if self._rope is None:
                self._rope = nn.RoPE(q.shape[-1])
            return self._rope(q), self._rope(k)
        raise ValueError(f"Unknown position kind: {self.kind}")
