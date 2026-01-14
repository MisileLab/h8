from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ecd.data.dataset import load_embeddings_dataset
from ecd.embed.cache import load_or_prepare_teacher_cache, write_meta
from ecd.index.ann import backend_available, brute_force_topk
from ecd.index.pipeline import PipelineConfig, run_one_stage, run_two_stage
from ecd.metrics.retrieval import compute_metrics
from ecd.plots.make_plots import make_plots
from ecd.sweeps.sweep import sweep
from ecd.train.trainer import load_checkpoint, train_student
from ecd.utils.config import load_config
from ecd.utils.device import resolve_device
from ecd.utils.io import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--dataset-config", type=str, default="configs/dataset/default.yaml"
    )
    parser.add_argument(
        "--model-config", type=str, default="configs/model/default.yaml"
    )
    parser.add_argument(
        "--train-config", type=str, default="configs/train/default.yaml"
    )
    parser.add_argument(
        "--index-config", type=str, default="configs/index/default.yaml"
    )
    parser.add_argument(
        "--sweep-config", type=str, default="configs/sweep/default.yaml"
    )
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_overrides = list(args.override) + [f"run_id={run_id}"]
    dataset_cfg = load_config([args.dataset_config], dataset_overrides)
    model_cfg = load_config([args.model_config])
    train_cfg = load_config([args.train_config], args.override)
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
    if model_cfg.get("in_dim") is None:
        model_cfg["in_dim"] = teacher_vectors.shape[1]
    output_dir = Path("results") / run_id / "checkpoints"
    log_dir = Path("results") / run_id / "train"
    train_student(
        teacher_vectors,
        teacher_topk,
        model_cfg,
        train_cfg,
        output_dir=output_dir,
        log_dir=log_dir,
    )
    best_checkpoint = output_dir / "student_best.pt"
    last_checkpoint = load_json(output_dir / "last_checkpoint.json")["path"]
    ckpt_path = str(best_checkpoint) if best_checkpoint.exists() else last_checkpoint
    device = resolve_device(train_cfg["device"])
    model, _ = load_checkpoint(ckpt_path)
    model = model.to(device)
    rep_vectors = (
        model(torch.tensor(teacher_vectors, dtype=torch.float32, device=device))
        .detach()
        .cpu()
        .numpy()
    )
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
    query_n = min(2000, teacher_vectors.shape[0])
    query_ids = np.arange(query_n)
    query_rep = rep_vectors[query_ids]
    if backend_available(pipeline_cfg.backend):
        if pipeline_cfg.use_two_stage:
            pred_teacher, _ = run_two_stage(
                rep_vectors,
                teacher_vectors,
                query_ids,
                k=100,
                candidate_n=pipeline_cfg.candidate_n,
                config=pipeline_cfg,
            )
        else:
            pred_teacher, _ = run_one_stage(
                rep_vectors, query_rep, k=100, config=pipeline_cfg
            )
        truth_teacher = teacher_topk[query_ids][:, :100]
        vs_teacher = compute_metrics(pred_teacher, truth_teacher, [10, 100])
        pred_student, _ = run_one_stage(
            rep_vectors, query_rep, k=100, config=pipeline_cfg
        )
        truth_student = brute_force_topk(
            rep_vectors, query_rep, k=100, metric=pipeline_cfg.metric
        )
        vs_student = compute_metrics(pred_student, truth_student, [10, 100])
        summary = {"vs_teacher": vs_teacher, "vs_student": vs_student}
        save_json(Path("results") / run_id / "student_metrics.json", summary)
        print(
            f"backend={pipeline_cfg.backend} metric={pipeline_cfg.metric} queries={query_n}"
        )
        print(f"k=10 vs_teacher recall={vs_teacher['recall@10']}")
        print(f"k=10 vs_student recall={vs_student['recall@10']}")
    else:
        save_json(
            Path("results") / run_id / "student_metrics.json",
            {"backend_status": "skipped"},
        )
        print("backend unavailable; skipping")
    hard_cfg = train_cfg["train"]["hard_negative"]
    rank_cfg = train_cfg["train"]["rank"]["multi_positive"]
    print(f"hard_negative={hard_cfg}")
    print(f"multi_positive={rank_cfg}")
    sweep_query_ids = np.arange(min(sweep_cfg["query_n"], teacher_vectors.shape[0]))
    for mode in sweep_cfg["representation_modes"]:
        mode_dir = Path("results") / run_id / mode
        sweep(
            run_id=run_id,
            representation_modes=[mode],
            rep_dims=sweep_cfg["rep_dims"],
            teacher_vectors=teacher_vectors,
            teacher_topk=teacher_topk,
            query_ids=sweep_query_ids,
            k_values=sweep_cfg["k_values"],
            pipeline_cfg=pipeline_cfg,
            output_dir=mode_dir,
            ckpt_path=ckpt_path,
        )
    make_plots(Path("results") / run_id)


if __name__ == "__main__":
    main()
