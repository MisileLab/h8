from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from ecd.index.ann import ANNConfig, ANNIndex, backend_available, brute_force_topk
from ecd.index.pipeline import PipelineConfig, run_one_stage, run_two_stage
from ecd.metrics.retrieval import compute_metrics
from ecd.train.trainer import load_checkpoint
from ecd.utils.io import ensure_dir, save_json
from ecd.utils.timing import now_ms


def _random_projection(
    vectors: np.ndarray, out_dim: int, normalize: bool
) -> np.ndarray:
    matrix = np.random.normal(size=(vectors.shape[1], out_dim)).astype(np.float32)
    projected = vectors @ matrix
    if normalize:
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / np.clip(norms, 1e-12, None)
    return projected.astype(np.float32)


def _pca_projection(vectors: np.ndarray, out_dim: int, normalize: bool) -> np.ndarray:
    model = PCA(n_components=out_dim, svd_solver="randomized")
    projected = model.fit_transform(vectors).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / np.clip(norms, 1e-12, None)
    return projected


def _student_projection(vectors: np.ndarray, ckpt_path: str) -> Tuple[np.ndarray, int]:
    model, cfg = load_checkpoint(ckpt_path)
    with torch.no_grad():
        output = (
            model(torch.tensor(vectors, dtype=torch.float32)).detach().cpu().numpy()
        )
    return output.astype(np.float32), cfg.out_dim


def build_representation(
    representation_mode: str,
    vectors: np.ndarray,
    rep_dim: Optional[int],
    ckpt_path: Optional[str],
    normalize: bool,
) -> Tuple[np.ndarray, Optional[int]]:
    if representation_mode == "trained_student":
        if ckpt_path is None:
            raise ValueError("ckpt_path required for trained_student")
        rep, out_dim = _student_projection(vectors, ckpt_path)
        return rep, out_dim
    if rep_dim is None:
        raise ValueError("rep_dim required for projection")
    if representation_mode == "random_projection":
        return _random_projection(vectors, rep_dim, normalize), rep_dim
    if representation_mode == "pca":
        return _pca_projection(vectors, rep_dim, normalize), rep_dim
    raise ValueError(f"Unknown representation_mode {representation_mode}")


def _evaluate_scope(
    scope: str,
    rep_vectors: np.ndarray,
    teacher_vectors: np.ndarray,
    teacher_topk: np.ndarray,
    query_ids: np.ndarray,
    k_values: List[int],
    pipeline_cfg: PipelineConfig,
) -> Tuple[List[Dict[str, float]], Optional[ANNIndex], float]:
    query_rep = rep_vectors[query_ids]
    output: List[Dict[str, float]] = []
    start = now_ms()
    if scope == "vs_teacher":
        if pipeline_cfg.use_two_stage:
            pred, index = run_two_stage(
                rep_vectors,
                teacher_vectors,
                query_ids,
                k=max(k_values),
                candidate_n=pipeline_cfg.candidate_n,
                config=pipeline_cfg,
            )
        else:
            pred, index = run_one_stage(
                rep_vectors, query_rep, k=max(k_values), config=pipeline_cfg
            )
        truth = teacher_topk[query_ids][:, : max(k_values)]
    else:
        ann_cfg = ANNConfig(
            backend=pipeline_cfg.backend,
            metric=pipeline_cfg.metric,
            force_normalize=pipeline_cfg.force_normalize,
            m=pipeline_cfg.m,
            ef_construction=pipeline_cfg.ef_construction,
            ef_search=pipeline_cfg.ef_search,
        )
        index = ANNIndex(ann_cfg)
        index.build(rep_vectors)
        pred = index.search(query_rep, k=max(k_values))
        truth = brute_force_topk(
            rep_vectors, query_rep, k=max(k_values), metric=pipeline_cfg.metric
        )
    latency_ms = now_ms() - start
    metrics = compute_metrics(pred, truth, k_values)
    output.append(metrics)
    return output, index, latency_ms


def estimate_qps(query_n: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return float(query_n / (latency_ms / 1000.0))


def sweep(
    run_id: str,
    representation_modes: List[str],
    rep_dims: List[int],
    teacher_vectors: np.ndarray,
    teacher_topk: np.ndarray,
    query_ids: np.ndarray,
    k_values: List[int],
    pipeline_cfg: PipelineConfig,
    output_dir: str | Path,
    ckpt_path: Optional[str],
) -> pl.DataFrame:
    rows: List[Dict[str, object]] = []
    for mode in tqdm(representation_modes, desc="representation modes"):
        dims = [None] if mode == "trained_student" else rep_dims
        for rep_dim in tqdm(dims, desc=f"{mode} dims", leave=False):
            rep_vectors, actual_dim = build_representation(
                mode,
                teacher_vectors,
                rep_dim,
                ckpt_path,
                normalize=pipeline_cfg.force_normalize,
            )
            rep_dim_value = int(actual_dim) if actual_dim is not None else None
            for scope in ["vs_teacher", "vs_student"]:
                if not backend_available(pipeline_cfg.backend):
                    row = {
                        "representation_mode": mode,
                        "scope": scope,
                        "d_out_requested": rep_dim
                        if mode != "trained_student"
                        else None,
                        "rep_dim": rep_dim_value,
                        "ckpt_path": ckpt_path if mode == "trained_student" else None,
                        "query_n": int(len(query_ids)),
                        "query_idx_first": int(query_ids[0]),
                        "query_idx_last": int(query_ids[-1]),
                        "teacher_topk_shape": str(teacher_topk.shape),
                        "backend": pipeline_cfg.backend,
                        "metric": pipeline_cfg.metric,
                        "use_two_stage": pipeline_cfg.use_two_stage,
                        "candidate_n": pipeline_cfg.candidate_n,
                        "latency_p50_ms": None,
                        "latency_p95_ms": None,
                        "qps": None,
                        "memory_bytes": None,
                        "backend_status": "skipped",
                    }
                    rows.append(row)
                    continue
                metrics_list, index, latency_ms = _evaluate_scope(
                    scope,
                    rep_vectors,
                    teacher_vectors,
                    teacher_topk,
                    query_ids,
                    k_values,
                    pipeline_cfg,
                )
                metrics = metrics_list[0]
                memory_bytes = index.memory_bytes() if index is not None else 0
                row = {
                    "representation_mode": mode,
                    "scope": scope,
                    "d_out_requested": rep_dim if mode != "trained_student" else None,
                    "rep_dim": rep_dim_value,
                    "ckpt_path": ckpt_path if mode == "trained_student" else None,
                    "query_n": int(len(query_ids)),
                    "query_idx_first": int(query_ids[0]),
                    "query_idx_last": int(query_ids[-1]),
                    "teacher_topk_shape": str(teacher_topk.shape),
                    "backend": pipeline_cfg.backend,
                    "metric": pipeline_cfg.metric,
                    "use_two_stage": pipeline_cfg.use_two_stage,
                    "candidate_n": pipeline_cfg.candidate_n,
                    "latency_p50_ms": latency_ms / max(1, len(query_ids)),
                    "latency_p95_ms": latency_ms / max(1, len(query_ids)),
                    "qps": estimate_qps(len(query_ids), latency_ms),
                    "memory_bytes": memory_bytes,
                    "backend_status": "ok",
                }
                row.update(metrics)
                rows.append(row)
    df = pl.DataFrame(rows)
    output_path = ensure_dir(Path(output_dir))
    df.write_parquet(output_path / "metrics.parquet")
    save_json(output_path / "summary.json", {"rows": len(rows)})
    return df


def select_best(
    df: pl.DataFrame, target_recall_at_10: float
) -> Optional[Dict[str, object]]:
    if df.is_empty():
        return None
    filtered = df.filter(
        (pl.col("scope") == "vs_teacher")
        & (pl.col("backend_status") == "ok")
        & (pl.col("recall@10") >= target_recall_at_10)
    )
    if filtered.is_empty():
        return None
    best = filtered.sort("latency_p95_ms").head(1).to_dicts()[0]
    return best
