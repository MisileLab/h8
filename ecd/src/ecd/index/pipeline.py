from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ecd.index.ann import ANNConfig, ANNIndex, brute_force_topk


@dataclass
class PipelineConfig:
    backend: str
    metric: str
    force_normalize: bool
    use_two_stage: bool
    candidate_n: int
    m: int
    ef_construction: int
    ef_search: int


def build_index(config: PipelineConfig) -> ANNIndex:
    ann_cfg = ANNConfig(
        backend=config.backend,
        metric=config.metric,
        force_normalize=config.force_normalize,
        m=config.m,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
    )
    return ANNIndex(ann_cfg)


def run_one_stage(
    rep_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    config: PipelineConfig,
) -> Tuple[np.ndarray, ANNIndex]:
    index = build_index(config)
    index.build(rep_vectors)
    return index.search(query_vectors, k), index


def run_two_stage(
    rep_vectors: np.ndarray,
    teacher_vectors: np.ndarray,
    query_ids: np.ndarray,
    k: int,
    candidate_n: int,
    config: PipelineConfig,
) -> Tuple[np.ndarray, ANNIndex]:
    index = build_index(config)
    index.build(rep_vectors)
    query_rep = rep_vectors[query_ids]
    candidates = index.search(query_rep, candidate_n)
    reranked = np.zeros((candidates.shape[0], k), dtype=np.int32)
    for i, candidate_ids in enumerate(candidates):
        candidate_vectors = teacher_vectors[candidate_ids]
        query = np.expand_dims(teacher_vectors[query_ids[i]], axis=0)
        scores = brute_force_topk(candidate_vectors, query, k=k, metric=config.metric)
        reranked[i] = candidate_ids[scores[0]]
    return reranked, index
