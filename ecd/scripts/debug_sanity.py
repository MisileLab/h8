from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from ecd.data.dataset import load_embeddings_dataset
from ecd.embed.cache import load_or_prepare_teacher_cache, write_meta
from ecd.index.ann import (
    ANNConfig,
    ANNIndex,
    backend_available,
    brute_force_topk,
    l2_normalize,
)
from ecd.metrics.retrieval import compute_metrics
from ecd.utils.config import load_config
from ecd.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--dataset-config", type=str, default="configs/dataset/default.yaml"
    )
    parser.add_argument(
        "--index-config", type=str, default="configs/index/default.yaml"
    )
    parser.add_argument(
        "--debug-config", type=str, default="configs/debug/default.yaml"
    )
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _collapse_stats(vectors: np.ndarray) -> dict:
    norms = np.linalg.norm(vectors, axis=1)
    rng = np.random.default_rng(0)
    pair_idx = rng.integers(0, vectors.shape[0], size=(min(5000, vectors.shape[0]), 2))
    cos = np.sum(
        l2_normalize(vectors[pair_idx[:, 0]]) * l2_normalize(vectors[pair_idx[:, 1]]),
        axis=1,
    )
    return {
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_p50": float(np.percentile(norms, 50)),
        "norm_p95": float(np.percentile(norms, 95)),
        "cos_mean": float(np.mean(cos)),
        "cos_std": float(np.std(cos)),
        "cos_p5": float(np.percentile(cos, 5)),
        "cos_p95": float(np.percentile(cos, 95)),
    }


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_overrides = list(args.override) + [f"run_id={run_id}"]
    dataset_cfg = load_config([args.dataset_config], dataset_overrides)
    index_cfg = load_config([args.index_config])
    debug_cfg = load_config([args.debug_config], args.override)
    cache_dir = dataset_cfg["cache_dir"].format(run_id=run_id)
    bundle = load_embeddings_dataset(
        name=dataset_cfg["name"],
        split=dataset_cfg["split"],
        embedding_fields_priority=dataset_cfg["embedding_fields_priority"],
        text_fields=dataset_cfg["text_fields"],
        max_rows=dataset_cfg["max_rows"],
    )
    teacher_vectors, _ = load_or_prepare_teacher_cache(
        bundle,
        cache_dir=cache_dir,
        k=dataset_cfg["teacher_topk"],
        metric=dataset_cfg["metric"],
    )
    write_meta(cache_dir, bundle)
    query_n = min(debug_cfg["query_n"], teacher_vectors.shape[0])
    query_ids = np.arange(query_n)
    rows = []
    if not backend_available(index_cfg["backend"]):
        output_dir = Path("results") / run_id / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "debug_summary.json", {"backend_status": "skipped"})
        return
    for normalize in debug_cfg["force_normalize_ab"]:
        ann_cfg = ANNConfig(
            backend=index_cfg["backend"],
            metric=index_cfg["metric"],
            force_normalize=normalize,
            m=index_cfg["m"],
            ef_construction=index_cfg["ef_construction"],
            ef_search=index_cfg["ef_search"],
        )
        index = ANNIndex(ann_cfg)
        index.build(teacher_vectors)
        query_vectors = teacher_vectors[query_ids]
        pred = index.search(query_vectors, k=max(debug_cfg["k_values"]))
        truth = brute_force_topk(
            teacher_vectors,
            query_vectors,
            k=max(debug_cfg["k_values"]),
            metric=index_cfg["metric"],
        )
        metrics = compute_metrics(pred, truth, debug_cfg["k_values"])
        rows.append({"force_normalize": normalize, **metrics})

    summary = _collapse_stats(teacher_vectors)
    output_dir = Path("results") / run_id / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "debug_summary.json", summary)
    pl.DataFrame(rows).write_parquet(output_dir / "debug_tables.parquet")


if __name__ == "__main__":
    main()
