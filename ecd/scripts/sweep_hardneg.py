from __future__ import annotations

import argparse
import copy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import polars as pl
import torch

from ecd.data.dataset import iter_batches, load_embeddings_dataset
from ecd.embed.cache import load_or_prepare_teacher_cache, write_meta
from ecd.index.ann import backend_available, brute_force_topk
from ecd.index.pipeline import PipelineConfig, run_one_stage
from ecd.metrics.retrieval import compute_metrics
from ecd.train.trainer import load_checkpoint, train_student
from ecd.utils.config import load_config
from ecd.utils.device import resolve_device
from ecd.utils.io import ensure_dir, save_json
from ecd.utils.seed import set_seed
from ecd.utils.timing import now_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sweep/hardneg.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _slug(value: object) -> str:
    return str(value).replace(".", "p")


def _encode(
    model: torch.nn.Module,
    vectors: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in iter_batches(vectors, batch_size):
            tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            outputs.append(model(tensor).detach().cpu().numpy())
    return np.vstack(outputs)


def _evaluate(
    model: torch.nn.Module,
    teacher_embeddings: np.ndarray,
    teacher_topk: np.ndarray,
    eval_cfg: Dict,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    if not backend_available(eval_cfg["backend"]):
        return {
            "vs_teacher_recall@10": None,
            "vs_teacher_ndcg@10": None,
            "vs_teacher_overlap@10": None,
            "vs_student_recall@10": None,
            "latency_p95_ms": None,
            "backend_status": "skipped",
        }
    rep_vectors = _encode(model, teacher_embeddings, batch_size, device)
    query_n = min(eval_cfg["query_n"], teacher_embeddings.shape[0])
    query_ids = np.arange(query_n)
    query_rep = rep_vectors[query_ids]
    pipeline_cfg = PipelineConfig(
        backend=eval_cfg["backend"],
        metric=eval_cfg["metric"],
        force_normalize=eval_cfg["force_normalize"],
        use_two_stage=eval_cfg["use_two_stage"],
        candidate_n=eval_cfg["candidate_n"],
        m=eval_cfg["m"],
        ef_construction=eval_cfg["ef_construction"],
        ef_search=eval_cfg["ef_search"],
    )
    start = now_ms()
    pred_teacher, _ = run_one_stage(
        rep_vectors, query_rep, k=max(eval_cfg["k_values"]), config=pipeline_cfg
    )
    latency = (now_ms() - start) / max(1, query_n)
    truth_teacher = teacher_topk[query_ids][:, : max(eval_cfg["k_values"])]
    vs_teacher = compute_metrics(pred_teacher, truth_teacher, eval_cfg["k_values"])
    truth_student = brute_force_topk(
        rep_vectors, query_rep, k=max(eval_cfg["k_values"]), metric=eval_cfg["metric"]
    )
    vs_student = compute_metrics(pred_teacher, truth_student, eval_cfg["k_values"])
    return {
        "vs_teacher_recall@10": float(vs_teacher.get("recall@10", 0.0)),
        "vs_teacher_ndcg@10": float(vs_teacher.get("ndcg@10", 0.0)),
        "vs_teacher_overlap@10": float(vs_teacher.get("overlap@10", 0.0)),
        "vs_student_recall@10": float(vs_student.get("recall@10", 0.0)),
        "latency_p95_ms": float(latency),
        "backend_status": "ok",
    }


def _run_combo(
    combo: Dict[str, object],
    stage: str,
    train_cfg: Dict,
    model_cfg: Dict,
    teacher_embeddings: np.ndarray,
    teacher_topk: np.ndarray,
    eval_cfg: Dict,
    run_root: Path,
    seed: int | None,
) -> Dict[str, object]:
    multi_positive_flag = cast(bool, combo["multi_positive"])
    num_positives = cast(int, combo["num_positives"])
    combo_id = f"mix{_slug(combo['mix_random_ratio'])}_tail{combo['tail_to']}_steps{combo['steps']}_mp{int(multi_positive_flag)}_p{num_positives}"
    run_dir = run_root / combo_id
    if stage:
        run_dir = run_dir / stage
    output_dir = run_dir / "checkpoints"
    log_dir = run_dir / "train"
    if seed is not None:
        set_seed(seed)
    local_train_cfg = copy.deepcopy(train_cfg)
    local_train_cfg["train"]["steps"] = combo["steps"]
    local_train_cfg["train"]["hard_negative"]["tail_from"] = combo["tail_from"]
    local_train_cfg["train"]["hard_negative"]["tail_to"] = combo["tail_to"]
    local_train_cfg["train"]["hard_negative"]["mix_random_ratio"] = combo[
        "mix_random_ratio"
    ]
    local_train_cfg["train"]["rank"]["multi_positive"]["enabled"] = combo[
        "multi_positive"
    ]
    local_train_cfg["train"]["rank"]["multi_positive"]["num_positives"] = combo[
        "num_positives"
    ]
    device = resolve_device(local_train_cfg["device"])
    model, _ = train_student(
        teacher_embeddings,
        teacher_topk,
        model_cfg,
        local_train_cfg,
        output_dir=output_dir,
        log_dir=log_dir,
    )
    best_checkpoint = output_dir / "student_best.pt"
    if best_checkpoint.exists():
        model, _ = load_checkpoint(best_checkpoint)
        model = model.to(device)
    metrics = _evaluate(
        model,
        teacher_embeddings,
        teacher_topk,
        eval_cfg,
        local_train_cfg["train"]["batch_size"],
        device,
    )
    row = {
        "mix_random_ratio": combo["mix_random_ratio"],
        "tail_from": combo["tail_from"],
        "tail_to": combo["tail_to"],
        "steps": combo["steps"],
        "multi_positive_enabled": combo["multi_positive"],
        "num_positives": combo["num_positives"],
        "stage": stage or "full",
        "run_path": str(run_dir),
    }
    row.update(metrics)
    return row


def _select_top(rows: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
    scored = [
        row
        for row in rows
        if row.get("vs_teacher_recall@10") is not None
        and row.get("backend_status") == "ok"
    ]

    def _score(row: Dict[str, object]) -> Tuple[float, float]:
        recall = cast(float, row["vs_teacher_recall@10"])
        latency = cast(float, row["latency_p95_ms"])
        return (-recall, latency)

    scored.sort(key=_score)
    return scored[:top_k]


def main() -> None:
    args = parse_args()
    sweep_cfg = load_config([args.config], args.override)
    run_id = (
        args.run_id
        or sweep_cfg.get("parent_run_id")
        or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    model_cfg = load_config([sweep_cfg["base"]["model_config"]])
    train_cfg = load_config([sweep_cfg["base"]["train_config"]])
    eval_cfg = sweep_cfg["eval"]
    dataset_overrides = list(args.override) + [f"run_id={run_id}"]
    dataset_cfg = load_config([sweep_cfg["base"]["dataset_config"]], dataset_overrides)
    cache_dir = dataset_cfg["cache_dir"].format(run_id=run_id)
    bundle = load_embeddings_dataset(
        name=dataset_cfg["name"],
        split=dataset_cfg["split"],
        embedding_fields_priority=dataset_cfg["embedding_fields_priority"],
        text_fields=dataset_cfg["text_fields"],
        max_rows=dataset_cfg["max_rows"],
    )
    teacher_embeddings, teacher_topk = load_or_prepare_teacher_cache(
        bundle,
        cache_dir=cache_dir,
        k=dataset_cfg["teacher_topk"],
        metric=dataset_cfg["metric"],
    )
    write_meta(cache_dir, bundle)
    if model_cfg.get("in_dim") is None:
        model_cfg["in_dim"] = teacher_embeddings.shape[1]
    run_root = ensure_dir(Path("results") / run_id / "hardneg_sweep")
    grid = list(
        product(
            sweep_cfg["sweep"]["mix_random_ratio"],
            sweep_cfg["sweep"]["tail_to"],
            sweep_cfg["sweep"]["steps"],
            sweep_cfg["sweep"]["multi_positive"],
            sweep_cfg["sweep"]["num_positives"],
        )
    )
    combos = [
        {
            "mix_random_ratio": float(mix),
            "tail_from": int(sweep_cfg["sweep"]["tail_from"]),
            "tail_to": int(tail),
            "steps": int(steps),
            "multi_positive": bool(mp),
            "num_positives": int(pos),
        }
        for mix, tail, steps, mp, pos in grid
    ]
    rows: List[Dict[str, object]] = []
    if sweep_cfg["search"]["two_stage"]:
        quick_rows = [
            _run_combo(
                combo,
                "quick",
                train_cfg,
                model_cfg,
                teacher_embeddings,
                teacher_topk,
                eval_cfg,
                run_root,
                args.seed,
            )
            for combo in combos
        ]
        rows.extend(quick_rows)
        top_rows = _select_top(quick_rows, sweep_cfg["search"]["top_k"])
        for row in top_rows:
            combo = {
                "mix_random_ratio": row["mix_random_ratio"],
                "tail_from": row["tail_from"],
                "tail_to": row["tail_to"],
                "steps": sweep_cfg["search"]["refine_steps"],
                "multi_positive": row["multi_positive_enabled"],
                "num_positives": row["num_positives"],
            }
            rows.append(
                _run_combo(
                    combo,
                    "refine",
                    train_cfg,
                    model_cfg,
                    teacher_embeddings,
                    teacher_topk,
                    eval_cfg,
                    run_root,
                    args.seed,
                )
            )
    else:
        for combo in combos:
            rows.append(
                _run_combo(
                    combo,
                    "",
                    train_cfg,
                    model_cfg,
                    teacher_embeddings,
                    teacher_topk,
                    eval_cfg,
                    run_root,
                    args.seed,
                )
            )
    df = pl.DataFrame(rows)
    df.write_parquet(run_root / "summary.parquet")
    best_rows = _select_top(rows, 3)
    for idx, row in enumerate(best_rows, start=1):
        print(
            f"{idx}. mix_random_ratio={row['mix_random_ratio']} tail_to={row['tail_to']} steps={row['steps']} multi_positive={row['multi_positive_enabled']} num_positives={row['num_positives']} recall@10={row['vs_teacher_recall@10']} latency_p95_ms={row['latency_p95_ms']}"
        )
    save_json(run_root / "summary.json", {"rows": len(rows)})


if __name__ == "__main__":
    main()
