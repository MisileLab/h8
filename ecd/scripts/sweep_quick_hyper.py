from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import sys
import traceback
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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


@dataclass
class EvalConfig:
    validation_queries: int
    backend: str
    metric: str
    candidate_n: int
    m: int
    ef_construction: int
    ef_search: int
    force_normalize: bool
    measure_latency: bool
    latency_queries: int
    latency_repeats: int


@dataclass
class Combo:
    mix_random_ratio: float
    tail_from: int
    tail_to: int
    num_positives: int
    steps: int
    lr: float
    lr_schedule: str
    warmup_steps: int
    amp: str
    batch_size: int


@dataclass
class SweepSettings:
    seed: int
    steps: int
    validation_queries: int
    backend: str
    metric: str
    target_vs_teacher_recall10: float
    grid_mix: List[float]
    grid_tail_to: List[int]
    grid_num_positives: List[int]
    grid_steps: List[int]
    grid_lr: List[float]
    grid_lr_schedule: List[str]
    grid_warmup_steps: List[int]
    grid_amp: List[str]
    grid_batch_size: List[int]
    tail_from: int
    measure_latency: bool
    latency_queries: int
    latency_repeats: int
    save_plots: bool


@dataclass
class BaseConfig:
    dataset_config: str
    model_config: str
    train_config: str


def _choose_best(
    rows: List[Dict[str, Any]],
    target_recall10: float,
) -> Optional[Dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("status") == "success"
        and row.get("vs_teacher_recall10") is not None
        and float(row["vs_teacher_recall10"]) >= target_recall10
    ]
    if not candidates:
        candidates = [
            row
            for row in rows
            if row.get("status") == "success"
            and row.get("vs_teacher_recall10") is not None
        ]
    if not candidates:
        return None

    def score(row: Dict[str, Any]) -> Tuple[float, float, float]:
        p95 = row.get("p95_latency_ms")
        p95_value = float(p95) if p95 is not None else float("inf")
        return (
            p95_value,
            -float(row["vs_teacher_recall10"]),
            float(row.get("train_wall_time_sec") or float("inf")),
        )

    candidates.sort(key=score)
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sweep/quick_hyper.yaml")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def _slug(value: float) -> str:
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


def _measure_latency(
    rep_vectors: np.ndarray,
    query_rep: np.ndarray,
    pipeline_cfg: PipelineConfig,
    repeats: int,
) -> Tuple[Optional[float], Optional[float]]:
    timings: List[float] = []
    for _ in range(repeats):
        start = now_ms()
        run_one_stage(rep_vectors, query_rep, k=10, config=pipeline_cfg)
        elapsed = (now_ms() - start) / max(1, query_rep.shape[0])
        timings.append(elapsed)
    if not timings:
        return None, None
    return (
        float(np.percentile(timings, 50)),
        float(np.percentile(timings, 95)),
    )


def _evaluate(
    model: torch.nn.Module,
    teacher_embeddings: np.ndarray,
    teacher_topk: np.ndarray,
    eval_cfg: EvalConfig,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not backend_available(eval_cfg.backend):
        return {
            "vs_teacher_recall10": None,
            "vs_teacher_ndcg10": None,
            "vs_teacher_overlap10": None,
            "vs_student_recall10": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "status": "skipped",
            "vs_teacher": None,
            "vs_student": None,
        }
    rep_vectors = _encode(model, teacher_embeddings, batch_size, device)
    query_n = min(eval_cfg.validation_queries, teacher_embeddings.shape[0])
    query_ids = np.arange(query_n)
    query_rep = rep_vectors[query_ids]
    pipeline_cfg = PipelineConfig(
        backend=eval_cfg.backend,
        metric=eval_cfg.metric,
        force_normalize=eval_cfg.force_normalize,
        use_two_stage=False,
        candidate_n=eval_cfg.candidate_n,
        m=eval_cfg.m,
        ef_construction=eval_cfg.ef_construction,
        ef_search=eval_cfg.ef_search,
    )
    pred_teacher, _ = run_one_stage(rep_vectors, query_rep, k=10, config=pipeline_cfg)
    truth_teacher = teacher_topk[query_ids][:, :10]
    vs_teacher = compute_metrics(pred_teacher, truth_teacher, [10])
    truth_student = brute_force_topk(
        rep_vectors, query_rep, k=10, metric=eval_cfg.metric
    )
    vs_student = compute_metrics(pred_teacher, truth_student, [10])
    p50_latency = None
    p95_latency = None
    if eval_cfg.measure_latency:
        latency_queries = min(eval_cfg.latency_queries, query_rep.shape[0])
        latency_rep = query_rep[:latency_queries]
        p50_latency, p95_latency = _measure_latency(
            rep_vectors,
            latency_rep,
            pipeline_cfg,
            eval_cfg.latency_repeats,
        )
    return {
        "vs_teacher_recall10": float(vs_teacher.get("recall@10", 0.0)),
        "vs_teacher_ndcg10": float(vs_teacher.get("ndcg@10", 0.0)),
        "vs_teacher_overlap10": float(vs_teacher.get("overlap@10", 0.0)),
        "vs_student_recall10": float(vs_student.get("recall@10", 0.0)),
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "status": "success",
        "vs_teacher": vs_teacher,
        "vs_student": vs_student,
    }


def _run_combo(
    combo: Combo,
    base_train_cfg: Dict,
    model_cfg: Dict,
    teacher_embeddings: np.ndarray,
    teacher_topk: np.ndarray,
    eval_cfg: EvalConfig,
    run_root: Path,
    seed: Optional[int],
) -> Dict[str, Any]:
    combo_id = "_".join(
        [
            f"mix{_slug(combo.mix_random_ratio)}",
            f"tail{combo.tail_to}",
            f"P{combo.num_positives}",
            f"S{combo.steps}",
            f"lr{_slug(combo.lr)}",
            f"sched{combo.lr_schedule}",
            f"warm{combo.warmup_steps}",
            f"amp{combo.amp}",
            f"bs{combo.batch_size}",
        ]
    )
    run_dir = run_root / "runs" / combo_id
    output_dir = run_dir / "checkpoints"
    log_dir = run_dir / "train"
    row = {
        "combo_id": combo_id,
        "seed": seed,
        "steps": combo.steps,
        "lr": combo.lr,
        "lr_schedule": combo.lr_schedule,
        "warmup_steps": combo.warmup_steps,
        "amp": combo.amp,
        "batch_size": combo.batch_size,
        "mix_random_ratio": combo.mix_random_ratio,
        "tail_from": combo.tail_from,
        "tail_to": combo.tail_to,
        "num_positives": combo.num_positives,
        "train_wall_time_sec": None,
        "vs_teacher_recall10": None,
        "vs_teacher_ndcg10": None,
        "vs_teacher_overlap10": None,
        "vs_student_recall10": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "status": "failed",
        "error": None,
        "error_reason": None,
        "run_path": str(run_dir),
    }
    try:
        if seed is not None:
            set_seed(seed)
        local_train_cfg = copy.deepcopy(base_train_cfg)
        local_train_cfg["train"]["steps"] = combo.steps
        local_train_cfg["train"]["train_steps"] = combo.steps
        local_train_cfg["train"]["lr"] = combo.lr
        local_train_cfg["train"]["lr_schedule"] = combo.lr_schedule
        local_train_cfg["train"]["warmup_steps"] = combo.warmup_steps
        local_train_cfg["train"]["amp"] = combo.amp
        local_train_cfg["train"]["batch_size"] = combo.batch_size
        local_train_cfg["train"]["hard_negative"]["tail_from"] = combo.tail_from
        local_train_cfg["train"]["hard_negative"]["tail_to"] = combo.tail_to
        local_train_cfg["train"]["hard_negative"]["mix_random_ratio"] = (
            combo.mix_random_ratio
        )
        local_train_cfg["train"]["rank"]["multi_positive"]["enabled"] = True
        local_train_cfg["train"]["rank"]["multi_positive"]["num_positives"] = (
            combo.num_positives
        )
        local_train_cfg["train"]["hard_negative"]["enabled"] = True
        local_train_cfg["train"]["hard_negative"]["mode"] = "teacher_tail"
        local_train_cfg["train"]["rank"]["kind"] = "info_nce"
        device = resolve_device(local_train_cfg["device"])
        started_at = datetime.now()
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
        ended_at = datetime.now()
        row["train_wall_time_sec"] = float((ended_at - started_at).total_seconds())
        metrics = _evaluate(
            model,
            teacher_embeddings,
            teacher_topk,
            eval_cfg,
            int(local_train_cfg["train"]["batch_size"]),
            device,
        )
        vs_teacher = metrics.get("vs_teacher")
        vs_student = metrics.get("vs_student")
        if isinstance(vs_teacher, dict):
            save_json(run_dir / "metrics_vs_teacher.json", vs_teacher)
        if isinstance(vs_student, dict):
            save_json(run_dir / "metrics_vs_student.json", vs_student)
        metrics.pop("vs_teacher", None)
        metrics.pop("vs_student", None)
        row.update(metrics)
        row["status"] = metrics.get("status", "success")
    except Exception as exc:
        ended_at = datetime.now()
        started_at_value = locals().get("started_at")
        if isinstance(started_at_value, datetime):
            row["train_wall_time_sec"] = float(
                (ended_at - started_at_value).total_seconds()
            )
        row["error"] = str(exc)
        row["error_reason"] = f"{type(exc).__name__}: {exc}"
        local_train_cfg_value = locals().get("local_train_cfg")
        device_value = locals().get("device")
        train_value = (
            local_train_cfg_value.get("train", {})
            if isinstance(local_train_cfg_value, dict)
            else {}
        )
        device_name = (
            str(local_train_cfg_value.get("device"))
            if isinstance(local_train_cfg_value, dict)
            else None
        )
        resolved_device = (
            device_value.type if isinstance(device_value, torch.device) else None
        )
        header = (
            "sweep combo failed\n"
            f"combo_id={combo_id} seed={seed}\n"
            f"device={device_name} resolved_device={resolved_device}\n"
            f"steps={combo.steps} amp={combo.amp} lr={combo.lr} schedule={combo.lr_schedule} warmup={combo.warmup_steps} bs={combo.batch_size}\n"
            f"save_every_steps={train_value.get('save_every_steps', train_value.get('save_every'))} "
            f"eval_every_steps={train_value.get('eval_every_steps', train_value.get('eval_every'))}\n"
        )
        print(header, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            ensure_dir(run_dir)
            (run_dir / "error.txt").write_text(header + "\n" + traceback.format_exc())
        except Exception:
            pass
        row["status"] = "failed"
    return row


def _print_top(rows: List[Dict[str, Any]], top_k: int) -> None:
    scored = [
        row
        for row in rows
        if row.get("status") == "success" and row.get("vs_teacher_recall10") is not None
    ]
    scored.sort(
        key=lambda r: (
            -float(r["vs_teacher_recall10"]),
            -float(r["vs_teacher_ndcg10"]),
            float(r["p95_latency_ms"])
            if r["p95_latency_ms"] is not None
            else float("inf"),
        )
    )
    print("rank combo_id recall10 ndcg10 p95_ms")
    for idx, row in enumerate(scored[:top_k], start=1):
        print(
            f"{idx} {row['combo_id']} {row['vs_teacher_recall10']} {row['vs_teacher_ndcg10']} {row['p95_latency_ms']}"
        )


def _plot_pareto(rows: List[Dict[str, Any]], output_path: Path) -> None:
    points = [
        row
        for row in rows
        if row.get("status") == "success" and row.get("p95_latency_ms") is not None
    ]
    if not points:
        return
    x = [float(row["p95_latency_ms"]) for row in points]
    y = [float(row["vs_teacher_recall10"]) for row in points]
    labels = [str(row["combo_id"]) for row in points]
    plt.figure()
    plt.scatter(x, y)
    for idx, label in enumerate(labels):
        plt.text(x[idx], y[idx], label, fontsize=8)
    plt.xlabel("p95_latency_ms")
    plt.ylabel("vs_teacher_recall10")
    plt.title("quick hyper sweep")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_sweep_config(
    cfg: Dict[str, Any],
) -> Tuple[SweepSettings, BaseConfig, EvalConfig]:
    sweep = cfg["sweep"]
    grid = sweep["grid"]
    fixed = sweep["fixed"]
    eval_cfg = sweep["eval"]
    output_cfg = sweep["output"]
    sweep_settings = SweepSettings(
        seed=int(sweep["seed"]),
        steps=int(sweep["steps"]),
        validation_queries=int(sweep["validation_queries"]),
        backend=str(sweep["backend"]),
        metric=str(sweep["metric"]),
        target_vs_teacher_recall10=float(sweep.get("target_vs_teacher_recall10", 0.70)),
        grid_mix=[float(x) for x in grid.get("mix_random_ratio", [0.5])],
        grid_tail_to=[int(x) for x in grid.get("tail_to", [50])],
        grid_num_positives=[int(x) for x in grid.get("num_positives", [4])],
        grid_steps=[int(x) for x in grid.get("steps", [int(sweep["steps"])])],
        grid_lr=[float(x) for x in grid.get("lr", [])]
        or [float(cfg.get("train", {}).get("lr", 1e-3))],
        grid_lr_schedule=[str(x) for x in grid.get("lr_schedule", ["constant"])],
        grid_warmup_steps=[int(x) for x in grid.get("warmup_steps", [0])],
        grid_amp=[str(x) for x in grid.get("amp", ["none"])],
        grid_batch_size=[int(x) for x in grid.get("batch_size", [])]
        or [int(cfg.get("train", {}).get("batch_size", 256))],
        tail_from=int(fixed["hard_negative"]["tail_from"]),
        measure_latency=bool(eval_cfg["measure_latency"]),
        latency_queries=int(eval_cfg["latency_queries"]),
        latency_repeats=int(eval_cfg["latency_repeats"]),
        save_plots=bool(output_cfg["save_plots"]),
    )
    base = cfg["base"]
    base_cfg = BaseConfig(
        dataset_config=str(base["dataset_config"]),
        model_config=str(base["model_config"]),
        train_config=str(base["train_config"]),
    )
    eval_config = EvalConfig(
        validation_queries=sweep_settings.validation_queries,
        backend=sweep_settings.backend,
        metric=sweep_settings.metric,
        candidate_n=200,
        m=16,
        ef_construction=200,
        ef_search=64,
        force_normalize=True,
        measure_latency=sweep_settings.measure_latency,
        latency_queries=sweep_settings.latency_queries,
        latency_repeats=sweep_settings.latency_repeats,
    )
    return sweep_settings, base_cfg, eval_config


def main() -> None:
    args = parse_args()
    cfg = load_config([args.config], args.override)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_settings, base_cfg, eval_cfg = _build_sweep_config(cfg)
    seed = args.seed if args.seed is not None else sweep_settings.seed
    dataset_overrides = list(args.override) + [f"run_id={run_id}"]
    dataset_cfg = load_config([base_cfg.dataset_config], dataset_overrides)
    model_cfg = load_config([base_cfg.model_config])
    train_cfg = load_config([base_cfg.train_config])
    bundle = load_embeddings_dataset(
        name=dataset_cfg["name"],
        split=dataset_cfg["split"],
        embedding_fields_priority=dataset_cfg["embedding_fields_priority"],
        text_fields=dataset_cfg["text_fields"],
        max_rows=dataset_cfg["max_rows"],
    )
    cache_dir = dataset_cfg["cache_dir"].format(run_id=run_id)
    teacher_embeddings, teacher_topk = load_or_prepare_teacher_cache(
        bundle,
        cache_dir=cache_dir,
        k=dataset_cfg["teacher_topk"],
        metric=dataset_cfg["metric"],
    )
    write_meta(cache_dir, bundle)
    if model_cfg.get("in_dim") is None:
        model_cfg["in_dim"] = teacher_embeddings.shape[1]
    run_root = ensure_dir(Path("results") / run_id / "quick_hyper")
    grid = list(
        product(
            sweep_settings.grid_mix,
            sweep_settings.grid_tail_to,
            sweep_settings.grid_num_positives,
            sweep_settings.grid_steps,
            sweep_settings.grid_lr,
            sweep_settings.grid_lr_schedule,
            sweep_settings.grid_warmup_steps,
            sweep_settings.grid_amp,
            sweep_settings.grid_batch_size,
        )
    )
    combos = [
        Combo(
            mix_random_ratio=float(mix),
            tail_from=sweep_settings.tail_from,
            tail_to=int(tail),
            num_positives=int(num_pos),
            steps=int(steps),
            lr=float(lr),
            lr_schedule=str(lr_schedule),
            warmup_steps=int(warmup_steps),
            amp=str(amp),
            batch_size=int(batch_size),
        )
        for mix, tail, num_pos, steps, lr, lr_schedule, warmup_steps, amp, batch_size in grid
    ]
    rows: List[Dict[str, Any]] = []
    for combo in combos:
        print(
            "start "
            f"mix={combo.mix_random_ratio} tail_to={combo.tail_to} P={combo.num_positives} "
            f"steps={combo.steps} lr={combo.lr} schedule={combo.lr_schedule} warmup={combo.warmup_steps} "
            f"amp={combo.amp} bs={combo.batch_size}"
        )
        row = _run_combo(
            combo,
            train_cfg,
            model_cfg,
            teacher_embeddings,
            teacher_topk,
            eval_cfg,
            run_root,
            seed,
        )
        rows.append(row)
        print(
            "done "
            f"{row['combo_id']} "
            f"recall10={row['vs_teacher_recall10']} ndcg10={row['vs_teacher_ndcg10']} "
            f"p95_ms={row['p95_latency_ms']} wall_s={row['train_wall_time_sec']} status={row['status']}"
        )
    df = pl.DataFrame(rows)
    df.write_parquet(run_root / "summary.parquet")
    df.write_csv(run_root / "summary.csv")

    best = _choose_best(rows, sweep_settings.target_vs_teacher_recall10)
    if best is not None:
        import json

        (run_root / "best.json").write_text(json.dumps(best, indent=2, sort_keys=True))

    _print_top(rows, 5)
    if sweep_settings.save_plots:
        figures_dir = ensure_dir(run_root / "figures")
        _plot_pareto(rows, figures_dir / "pareto_vs_teacher.png")


if __name__ == "__main__":
    main()
