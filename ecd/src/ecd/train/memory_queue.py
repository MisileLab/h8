from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class MemoryQueueConfig:
    size: int
    sample_size: int
    device: str


class TeacherEmbeddingQueue:
    def __init__(
        self,
        embedding_dim: int,
        size: int,
        device: torch.device,
        use_cpu_storage: bool = False,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.size = int(size)
        self.device = device
        self.use_cpu_storage = bool(use_cpu_storage)

        storage_device = torch.device("cpu") if use_cpu_storage else device
        self._buffer = torch.empty(
            (self.size, self.embedding_dim),
            dtype=torch.float32,
            device=storage_device,
        )
        self._ptr = 0
        self._full = False

    @property
    def num_filled(self) -> int:
        return self.size if self._full else self._ptr

    def reset(self) -> None:
        self._ptr = 0
        self._full = False

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embeddings must be (N,{self.embedding_dim}) got {tuple(embeddings.shape)}"
            )

        # Always detach and keep float32.
        emb = embeddings.detach().to(dtype=torch.float32)
        if self.use_cpu_storage:
            emb = emb.to("cpu")

        n = int(emb.shape[0])
        if n <= 0:
            return

        if n >= self.size:
            # keep only last 'size' entries.
            self._buffer.copy_(emb[-self.size :])
            self._ptr = 0
            self._full = True
            return

        end = self._ptr + n
        if end <= self.size:
            self._buffer[self._ptr : end].copy_(emb)
        else:
            first = self.size - self._ptr
            self._buffer[self._ptr :].copy_(emb[:first])
            self._buffer[: end - self.size].copy_(emb[first:])

        self._ptr = end % self.size
        if end >= self.size:
            self._full = True

    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")

        filled = self.num_filled
        if filled <= 0:
            raise RuntimeError("memory queue is empty")

        device = device or self.device

        idx = torch.randint(
            low=0,
            high=filled,
            size=(n,),
            device=torch.device("cpu") if self.use_cpu_storage else device,
            generator=generator,
        )

        if self.use_cpu_storage:
            out = self._buffer[idx].to(device)
        else:
            out = self._buffer[idx]

        return out


def resolve_queue_storage(
    prefer_device_storage: bool,
    queue_size: int,
    embedding_dim: int,
    device: torch.device,
) -> bool:
    if not prefer_device_storage:
        return True

    if device.type != "cuda":
        # On cpu/mps we just keep it on the same device (mps memory is unified; cpu is cpu).
        return False

    # Rough estimate: float32 bytes.
    bytes_needed = int(queue_size) * int(embedding_dim) * 4
    # If CUDA reports 0 (rare), try device storage.
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        # Keep headroom.
        if bytes_needed > int(free_bytes * 0.5):
            return True
        return False
    except Exception:
        return False


def l2_normalize_np(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vectors / norms).astype(np.float32)
