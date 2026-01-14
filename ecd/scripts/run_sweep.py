from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from ecd.data.dataset import load_embeddings_dataset
from ecd.embed.cache import load_or_prepare_teacher_cache, write_meta
from ecd.index.pipeline import PipelineConfig
from ecd.sweeps.sweep import select_best, sweep
from ecd.utils.config import load_config
from ecd.utils.io import load_json, save_json


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
        "--sweep-config", type=str, default="configs/sweep/default.yaml"
    )
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _resolve_checkpoint(run_id: str, ckpt_path: str | None) -> str | None:
    if ckpt_path is not None:
        return ckpt_path
    last_checkpoint = Path("results") / run_id / "checkpoints" / "last_checkpoint.json"
    if last_checkpoint.exists():
        payload = load_json(last_checkpoint)
        return payload.get("path")
    return None


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_overrides = list(args.override) + [f"run_id={run_id}"]
    dataset_cfg = load_config([args.dataset_config], dataset_overrides)
    index_cfg = load_config([args.index_config])
    sweep_cfg = load_config([args.sweep_config], args.override)
    cache_dir = dataset_cfg["cache_dir"].format(run_id=run_id)
    bundle = load_embeddings_dataset(
        name=dataset_cfg["name"],
        split=dataset_cfg["split"],
        embedding_fields_priority=dataset_cfg["embedding_fields_priority"],
        text_fields=dataset_cfg["text_fields"],
        max_rows=dataset_cfg["max_rows"],
    )
    teacher_vectors, teacher_topk = load_or_prepare_teacher_cache(
        bundle,
        cache_dir=cache_dir,
        k=dataset_cfg["teacher_topk"],
        metric=dataset_cfg["metric"],
    )
    write_meta(cache_dir, bundle)
    pipeline_cfg = PipelineConfig(
        backend=index_cfg["backend"],
        metric=index_cfg["metric"],
        force_normalize=index_cfg["force_normalize"],
        use_two_stage=index_cfg["use_two_stage"],
        candidate_n=index_cfg["candidate_n"],
        m=index_cfg["m"],
        ef_construction=index_cfg["ef_construction"],
        ef_search=index_cfg["ef_search"],
    )
    query_ids = np.arange(min(sweep_cfg["query_n"], teacher_vectors.shape[0]))
    ckpt_path = _resolve_checkpoint(run_id, args.ckpt_path)
    for mode in sweep_cfg["representation_modes"]:
        mode_dir = Path("results") / run_id / mode
        df = sweep(
            run_id=run_id,
            representation_modes=[mode],
            rep_dims=sweep_cfg["rep_dims"],
            teacher_vectors=teacher_vectors,
            teacher_topk=teacher_topk,
            query_ids=query_ids,
            k_values=sweep_cfg["k_values"],
            pipeline_cfg=pipeline_cfg,
            output_dir=mode_dir,
            ckpt_path=ckpt_path,
        )
        target = (
            sweep_cfg["select"]["strict_target_recall_at_10"]
            if sweep_cfg["select"]["use_strict"]
            else sweep_cfg["select"]["target_recall_at_10"]
        )
        best = select_best(df, target)
        save_json(mode_dir / "summary.json", {"best": best, "target_recall@10": target})


if __name__ == "__main__":
    main()
