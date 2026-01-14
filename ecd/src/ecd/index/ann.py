from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import hnswlib
except Exception:
    hnswlib = None


def backend_available(backend: str) -> bool:
    if backend == "hnsw":
        return hnswlib is not None
    if backend == "none":
        return True
    return False


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def brute_force_topk(
    vectors: np.ndarray, queries: np.ndarray, k: int, metric: str
) -> np.ndarray:
    if metric == "cosine":
        vectors = l2_normalize(vectors)
        queries = l2_normalize(queries)
        scores = queries @ vectors.T
    else:
        scores = queries @ vectors.T
    topk = np.argpartition(scores, -k, axis=1)[:, -k:]
    topk_scores = np.take_along_axis(scores, topk, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    return np.take_along_axis(topk, order, axis=1).astype(np.int32)


@dataclass
class ANNConfig:
    backend: str
    metric: str
    force_normalize: bool
    m: int
    ef_construction: int
    ef_search: int


class ANNIndex:
    def __init__(self, config: ANNConfig) -> None:
        self.config = config
        self.index = None
        self.vectors: Optional[np.ndarray] = None

    def build(self, vectors: np.ndarray) -> None:
        if self.config.force_normalize and self.config.metric == "cosine":
            vectors = l2_normalize(vectors)
        self.vectors = vectors.astype(np.float32)
        if self.config.backend == "none":
            return
        if self.config.backend != "hnsw":
            raise ValueError(f"Unknown backend {self.config.backend}")
        if hnswlib is None:
            raise RuntimeError("hnswlib not available")
        space = "cosine" if self.config.metric == "cosine" else "ip"
        index = hnswlib.Index(space=space, dim=self.vectors.shape[1])
        index.init_index(
            max_elements=self.vectors.shape[0],
            ef_construction=self.config.ef_construction,
            M=self.config.m,
        )
        index.add_items(self.vectors, np.arange(self.vectors.shape[0]))
        index.set_ef(self.config.ef_search)
        self.index = index

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        if self.vectors is None:
            raise RuntimeError("Index not built")
        if self.config.force_normalize and self.config.metric == "cosine":
            queries = l2_normalize(queries)
        if self.config.backend == "none":
            return brute_force_topk(
                self.vectors, queries, k=k, metric=self.config.metric
            )
        if self.index is None:
            raise RuntimeError("Index not built")
        labels, _ = self.index.knn_query(queries.astype(np.float32), k=k)
        return labels.astype(np.int32)

    def memory_bytes(self) -> int:
        if self.vectors is None:
            return 0
        vector_bytes = int(self.vectors.size * self.vectors.itemsize)
        if self.config.backend != "hnsw":
            return vector_bytes
        link_bytes = int(self.vectors.shape[0] * self.config.m * 8)
        return vector_bytes + link_bytes
