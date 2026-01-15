from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, ContextManager, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ecd.data.dataset import iter_batches
from ecd.index.ann import backend_available, brute_force_topk
from ecd.index.pipeline import PipelineConfig, run_one_stage, run_two_stage
from ecd.losses.distill import distill_loss
from ecd.losses.mix import LossMixConfig, mix_losses
from ecd.losses.rank import info_nce, info_nce_multi
from ecd.losses.struct import distortion_loss
from ecd.metrics.retrieval import compute_metrics
from ecd.models.student import StudentConfig, build_student
from ecd.train.memory_queue import TeacherEmbeddingQueue, resolve_queue_storage
from ecd.train.track_b import (
    TrackBConfig,
    build_track_b_config,
    build_candidate_vectors,
    candidates_from_topk_and_queue,
    listwise_kl_distill,
    reshape_queue_samples,
)
from ecd.utils.device import (
    DynamicBatchConfig,
    DynamicBatchSizer,
    clear_memory_cache,
    estimate_encode_batch_size,
    get_vram_info,
    is_oom_error,
    resolve_device,
)
from ecd.utils.io import ensure_dir, save_json
from ecd.utils.seed import set_seed


@dataclass
class HardNegativeConfig:
    enabled: bool
    mode: str
    tail_from: int
    tail_to: int
    mix_random_ratio: float


@dataclass
class MultiPositiveConfig:
    enabled: bool
    num_positives: int


@dataclass
class RankConfig:
    kind: str
    multi_positive: MultiPositiveConfig


@dataclass
class EvalConfig:
    query_n: int
    k_values: List[int]
    backend: str
    metric: str
    force_normalize: bool
    use_two_stage: bool
    candidate_n: int
    m: int
    ef_construction: int
    ef_search: int
    eval_every: int


@dataclass
class TrainConfig:
    seed: int
    device: str
    steps: int
    batch_size: int
    lr: float
    weight_decay: float
    lr_schedule: str
    warmup_steps: int
    amp: str
    resume_path: Optional[str]
    log_every: int
    save_every: int
    use_distill: bool
    use_rank: bool
    use_struct: bool
    distill_mode: str
    rank_temperature: float
    rank_kind: str
    multi_positive: MultiPositiveConfig
    loss_mix: LossMixConfig
    hard_negative: HardNegativeConfig
    track_b: TrackBConfig
    dynamic_batch: DynamicBatchConfig


def build_train_config(train_cfg: Dict) -> Tuple[TrainConfig, EvalConfig]:
    train = train_cfg["train"]
    loss = train_cfg["loss"]
    hard_cfg = train["hard_negative"]
    rank_cfg = train["rank"]
    multi_cfg = rank_cfg["multi_positive"]
    eval_cfg = train_cfg["eval"]
    steps = int(train.get("train_steps", train.get("steps")))
    save_every_raw = train.get("save_every_steps", train.get("save_every", 500))
    save_every = int(save_every_raw) if save_every_raw is not None else 0
    eval_every_raw = train.get("eval_every_steps", train.get("eval_every", 500))
    eval_every = int(eval_every_raw) if eval_every_raw is not None else 0
    log_every = int(train.get("log_every_steps", train.get("log_every", 100)))
    track_b_cfg = build_track_b_config(train_cfg)

    # Parse dynamic batch configuration
    dyn_batch_cfg = train.get("dynamic_batch", {})
    batch_size_raw = train.get("batch_size", 256)
    dynamic_batch_config = DynamicBatchConfig(
        enabled=bool(dyn_batch_cfg.get("enabled", False)),
        initial_batch_size=(
            int(batch_size_raw) if batch_size_raw not in ("auto", None) else None
        ),
        min_batch_size=int(dyn_batch_cfg.get("min_batch_size", 16)),
        max_batch_size=int(dyn_batch_cfg.get("max_batch_size", 2048)),
        memory_fraction=float(dyn_batch_cfg.get("memory_fraction", 0.6)),
        oom_retry_enabled=bool(dyn_batch_cfg.get("oom_retry_enabled", True)),
        oom_reduction_factor=float(dyn_batch_cfg.get("oom_reduction_factor", 0.7)),
        max_oom_retries=int(dyn_batch_cfg.get("max_oom_retries", 5)),
        warmup_steps=int(dyn_batch_cfg.get("warmup_steps", 10)),
        growth_factor=float(dyn_batch_cfg.get("growth_factor", 1.1)),
        growth_check_interval=int(dyn_batch_cfg.get("growth_check_interval", 50)),
    )

    train_config = TrainConfig(
        seed=int(train_cfg["seed"]),
        device=str(train_cfg["device"]),
        steps=steps,
        batch_size=int(train["batch_size"]) if train.get("batch_size") not in ("auto", None) else 256,
        lr=float(train["lr"]),
        weight_decay=float(train["weight_decay"]),
        lr_schedule=str(train.get("lr_schedule", "constant")),
        warmup_steps=int(train.get("warmup_steps", 0)),
        amp=str(train.get("amp", "none")),
        resume_path=train.get("resume_path"),
        log_every=log_every,
        save_every=save_every,
        use_distill=bool(loss["distill"]),
        use_rank=bool(loss["rank"]),
        use_struct=bool(loss["struct"]),
        distill_mode=str(loss["distill_mode"]),
        rank_temperature=float(loss["rank_temperature"]),
        rank_kind=str(rank_cfg["kind"]),
        multi_positive=MultiPositiveConfig(
            enabled=bool(multi_cfg["enabled"]),
            num_positives=int(multi_cfg["num_positives"]),
        ),
        loss_mix=LossMixConfig(
            alpha=float(loss["alpha"]),
            beta=float(loss["beta"]),
            gamma=float(loss["gamma"]),
            warmup_frac=float(loss["warmup_frac"]),
        ),
        hard_negative=HardNegativeConfig(
            enabled=bool(hard_cfg["enabled"]),
            mode=str(hard_cfg["mode"]),
            tail_from=int(hard_cfg["tail_from"]),
            tail_to=int(hard_cfg["tail_to"]),
            mix_random_ratio=float(hard_cfg["mix_random_ratio"]),
        ),
        track_b=track_b_cfg,
        dynamic_batch=dynamic_batch_config,
    )
    eval_config = EvalConfig(
        query_n=int(eval_cfg["query_n"]),
        k_values=list(eval_cfg["k_values"]),
        backend=str(eval_cfg["backend"]),
        metric=str(eval_cfg["metric"]),
        force_normalize=bool(eval_cfg["force_normalize"]),
        use_two_stage=bool(eval_cfg["use_two_stage"]),
        candidate_n=int(eval_cfg["candidate_n"]),
        m=int(eval_cfg["m"]),
        ef_construction=int(eval_cfg["ef_construction"]),
        ef_search=int(eval_cfg["ef_search"]),
        eval_every=eval_every,
    )
    return train_config, eval_config


def _norm_stats(vectors: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "p50": float(np.percentile(norms, 50)),
        "p95": float(np.percentile(norms, 95)),
    }


def _cosine_stats(
    vectors: np.ndarray, sample_size: int, rng: np.random.Generator
) -> Dict[str, float]:
    if vectors.shape[0] < 2:
        return {"mean": 0.0, "std": 0.0, "p5": 0.0, "p95": 0.0}
    count = min(sample_size, vectors.shape[0] * 2)
    pair_idx = rng.integers(0, vectors.shape[0], size=(count, 2))
    left = vectors[pair_idx[:, 0]]
    right = vectors[pair_idx[:, 1]]
    left = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-12, None)
    right = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-12, None)
    cos = np.sum(left * right, axis=1)
    return {
        "mean": float(np.mean(cos)),
        "std": float(np.std(cos)),
        "p5": float(np.percentile(cos, 5)),
        "p95": float(np.percentile(cos, 95)),
    }


def _tensor_percentiles(values: torch.Tensor, percentiles: List[float]) -> List[float]:
    probs = torch.tensor([p / 100.0 for p in percentiles], device=values.device)
    try:
        quantiles = torch.quantile(values, probs)
        return [float(value.item()) for value in quantiles]
    except Exception:
        array = values.detach().cpu().numpy()
        return [float(np.percentile(array, p)) for p in percentiles]


def _norm_stats_torch(vectors: torch.Tensor) -> Dict[str, float]:
    values = torch.linalg.norm(vectors.detach(), dim=1)
    p50, p95 = _tensor_percentiles(values, [50, 95])
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
        "p50": p50,
        "p95": p95,
    }


def _cosine_stats_torch(vectors: torch.Tensor, sample_size: int) -> Dict[str, float]:
    if vectors.shape[0] < 2:
        return {"mean": 0.0, "std": 0.0, "p5": 0.0, "p95": 0.0}
    vectors = vectors.detach()
    count = min(sample_size, int(vectors.shape[0]) * 2)
    pair_idx = torch.randint(0, vectors.shape[0], (count, 2), device=vectors.device)
    left = vectors[pair_idx[:, 0]]
    right = vectors[pair_idx[:, 1]]
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    cos = torch.sum(left * right, dim=1)
    p5, p95 = _tensor_percentiles(cos, [5, 95])
    return {
        "mean": float(cos.mean().item()),
        "std": float(cos.std().item()),
        "p5": p5,
        "p95": p95,
    }


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    schedule: str,
) -> LambdaLR:
    warmup_steps = min(max(int(warmup_steps), 0), max(total_steps, 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return max(0.0, 1.0 - progress)
        if schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        raise ValueError(f"Unknown lr_schedule {schedule}")

    return LambdaLR(optimizer, lr_lambda)


def _build_amp_context(
    amp_mode: str, device: torch.device
) -> Tuple[ContextManager[object], Optional[torch.cuda.amp.GradScaler]]:
    if amp_mode == "none":
        return nullcontext(), None
    if amp_mode == "fp16":
        if device.type == "cuda":
            return (
                cast(
                    ContextManager[object],
                    torch.autocast(device_type="cuda", dtype=torch.float16),
                ),
                torch.cuda.amp.GradScaler(),
            )
        if device.type == "mps":
            return (
                cast(
                    ContextManager[object],
                    torch.autocast(device_type="mps", dtype=torch.float16),
                ),
                None,
            )
        raise RuntimeError("fp16 AMP requested but not supported on this device")
    if amp_mode == "bf16":
        if device.type in {"cuda", "cpu"}:
            return (
                cast(
                    ContextManager[object],
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
                ),
                None,
            )
        raise RuntimeError(
            f"bf16 AMP not supported on device={device.type}; use amp=none or amp=fp16 (if supported)."
        )
    raise ValueError(f"Unknown amp mode {amp_mode}")


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> Tuple[int, bool]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer_state(optimizer, device)
    has_scheduler = False
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
        has_scheduler = True
    if scaler is not None and "scaler_state" in payload:
        scaler.load_state_dict(payload["scaler_state"])
    return int(payload.get("step", 0)), has_scheduler


def _sample_positive_indices(
    teacher_topk: torch.Tensor,
    anchor_indices: torch.Tensor,
    positive_k: int,
    num_positives: int,
) -> torch.Tensor:
    pool = teacher_topk[anchor_indices, :positive_k]
    num_pos = max(1, min(num_positives, pool.shape[1]))
    choices = torch.randint(
        0,
        pool.shape[1],
        (anchor_indices.shape[0], num_pos),
        device=anchor_indices.device,
    )
    return torch.gather(pool, 1, choices)


def _sample_hard_negatives(
    teacher_topk: torch.Tensor,
    anchor_indices: torch.Tensor,
    tail_from: int,
    tail_to: int,
    num_samples: int,
) -> Optional[torch.Tensor]:
    k = teacher_topk.shape[1]
    end = min(tail_to, k)
    start = min(tail_from, end)
    if start >= end:
        return None
    pool = teacher_topk[anchor_indices, start:end]
    if pool.shape[1] == 0:
        return None
    choices = torch.randint(
        0,
        pool.shape[1],
        (anchor_indices.shape[0], num_samples),
        device=anchor_indices.device,
    )
    return torch.gather(pool, 1, choices)


def _sample_random_negatives(
    num_items: int,
    anchor_indices: torch.Tensor,
    positive_indices: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    negatives = torch.randint(
        0,
        num_items,
        (anchor_indices.shape[0], num_samples),
        device=anchor_indices.device,
    )
    if positive_indices.ndim == 1:
        positive_indices = positive_indices[:, None]
    invalid = (negatives == anchor_indices[:, None]) | (
        negatives[..., None] == positive_indices[:, None, :]
    ).any(dim=2)
    while invalid.any():
        negatives[invalid] = torch.randint(
            0,
            num_items,
            (int(invalid.sum().item()),),
            device=anchor_indices.device,
        )
        invalid = (negatives == anchor_indices[:, None]) | (
            negatives[..., None] == positive_indices[:, None, :]
        ).any(dim=2)
    return negatives


def _mix_negatives(
    hard_negatives: Optional[torch.Tensor],
    random_negatives: torch.Tensor,
    mix_random_ratio: float,
) -> torch.Tensor:
    if hard_negatives is None:
        return random_negatives
    if mix_random_ratio <= 0:
        return hard_negatives
    if mix_random_ratio >= 1:
        return random_negatives
    mask = (
        torch.rand(hard_negatives.shape, device=hard_negatives.device)
        < mix_random_ratio
    )
    return torch.where(mask, random_negatives, hard_negatives)


def _ensure_valid_negatives(
    negatives: torch.Tensor,
    num_items: int,
    anchor_indices: torch.Tensor,
    positive_indices: torch.Tensor,
) -> torch.Tensor:
    if positive_indices.ndim == 1:
        positive_indices = positive_indices[:, None]
    invalid = (negatives == anchor_indices[:, None]) | (
        negatives[..., None] == positive_indices[:, None, :]
    ).any(dim=2)
    while invalid.any():
        negatives[invalid] = torch.randint(
            0,
            num_items,
            (int(invalid.sum().item()),),
            device=anchor_indices.device,
        )
        invalid = (negatives == anchor_indices[:, None]) | (
            negatives[..., None] == positive_indices[:, None, :]
        ).any(dim=2)
    return negatives


def _encode_vectors(
    model: nn.Module,
    vectors: np.ndarray,
    batch_size: Optional[int],
    device: torch.device,
    output_dim: Optional[int] = None,
) -> np.ndarray:
    was_training = model.training
    model.eval()

    if batch_size is None:
        in_dim = vectors.shape[1]
        out_dim = output_dim if output_dim is not None else in_dim
        batch_size = estimate_encode_batch_size(
            embedding_dim=in_dim,
            output_dim=out_dim,
            device=device,
        )

    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in iter_batches(vectors, batch_size):
            tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            outputs.append(model(tensor).detach().cpu().numpy())
    if was_training:
        model.train()
    return np.vstack(outputs)


def _evaluate(
    model: nn.Module,
    teacher_embeddings: np.ndarray,
    teacher_topk: np.ndarray,
    eval_cfg: EvalConfig,
    device: torch.device,
    batch_size: Optional[int] = None,
    output_dim: Optional[int] = None,
) -> Dict[str, object]:
    if not backend_available(eval_cfg.backend):
        return {
            "eval_status": None,
            "vs_teacher_recall@10": None,
            "vs_teacher_ndcg@10": None,
            "vs_teacher_overlap@10": None,
            "vs_student_recall@10": None,
        }
    rep_vectors = _encode_vectors(
        model, teacher_embeddings, batch_size, device, output_dim
    )
    query_n = min(eval_cfg.query_n, teacher_embeddings.shape[0])
    query_ids = np.arange(query_n)
    query_rep = rep_vectors[query_ids]
    pipeline_cfg = PipelineConfig(
        backend=eval_cfg.backend,
        metric=eval_cfg.metric,
        force_normalize=eval_cfg.force_normalize,
        use_two_stage=eval_cfg.use_two_stage,
        candidate_n=eval_cfg.candidate_n,
        m=eval_cfg.m,
        ef_construction=eval_cfg.ef_construction,
        ef_search=eval_cfg.ef_search,
    )
    if pipeline_cfg.use_two_stage:
        pred_teacher, _ = run_two_stage(
            rep_vectors,
            teacher_embeddings,
            query_ids,
            k=max(eval_cfg.k_values),
            candidate_n=pipeline_cfg.candidate_n,
            config=pipeline_cfg,
        )
    else:
        pred_teacher, _ = run_one_stage(
            rep_vectors, query_rep, k=max(eval_cfg.k_values), config=pipeline_cfg
        )
    truth_teacher = teacher_topk[query_ids][:, : max(eval_cfg.k_values)]
    vs_teacher = compute_metrics(pred_teacher, truth_teacher, eval_cfg.k_values)
    pred_student, _ = run_one_stage(
        rep_vectors, query_rep, k=max(eval_cfg.k_values), config=pipeline_cfg
    )
    truth_student = brute_force_topk(
        rep_vectors, query_rep, k=max(eval_cfg.k_values), metric=eval_cfg.metric
    )
    vs_student = compute_metrics(pred_student, truth_student, eval_cfg.k_values)
    return {
        "eval_status": "ok",
        "vs_teacher_recall@10": float(vs_teacher.get("recall@10", 0.0)),
        "vs_teacher_ndcg@10": float(vs_teacher.get("ndcg@10", 0.0)),
        "vs_teacher_overlap@10": float(vs_teacher.get("overlap@10", 0.0)),
        "vs_student_recall@10": float(vs_student.get("recall@10", 0.0)),
    }


def _plot_lines(
    steps: List[int],
    series: List[List[float]],
    labels: List[str],
    title: str,
    path: Path,
) -> None:
    if not steps:
        return
    plt.figure()
    for values, label in zip(series, labels):
        plt.plot(steps, values, label=label)
    plt.title(title)
    plt.xlabel("step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@dataclass(frozen=True)
class _LogNormalizationDiagnostics:
    mixed_type_columns: Dict[str, Dict[type, int]]


def _normalize_log_rows_for_polars(
    rows: List[Dict[str, object]],
) -> Tuple[
    List[Dict[str, object]],
    _LogNormalizationDiagnostics,
    object,
]:
    if not rows:
        from polars._typing import SchemaDict  # pyright: ignore[reportMissingImports]

        return (
            [],
            _LogNormalizationDiagnostics(mixed_type_columns={}),
            cast(SchemaDict, {}),
        )

    all_keys: List[str] = sorted({k for r in rows for k in r.keys()})

    type_counts: Dict[str, Counter[type]] = defaultdict(Counter)
    for r in rows:
        for k in all_keys:
            v = r.get(k)
            if v is None:
                continue
            type_counts[k][type(v)] += 1

    mixed: Dict[str, Dict[type, int]] = {}
    stringify_cols: set[str] = set()
    schema_overrides: Dict[str, object] = {}
    for k, cnt in type_counts.items():
        if len(cnt) > 1:
            mixed[k] = dict(cnt)
            stringify_cols.add(k)
            schema_overrides[k] = pl.Utf8()

    normalized: List[Dict[str, object]] = []
    for r in rows:
        nr: Dict[str, object] = {}
        for k in all_keys:
            v = r.get(k)
            if v is None:
                nr[k] = None
            elif k in stringify_cols:
                nr[k] = str(v)
            else:
                nr[k] = v
        normalized.append(nr)

    schema_overrides_typed: Dict[str, object] = dict(schema_overrides)
    return (
        normalized,
        _LogNormalizationDiagnostics(mixed_type_columns=mixed),
        schema_overrides_typed,
    )


def _write_log_rows_jsonl(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _make_train_plots(logs: pl.DataFrame, output_dir: Path) -> None:
    output_path = ensure_dir(output_dir)
    loss_df = logs.filter(pl.col("loss_total").is_not_null())
    steps = loss_df["step"].to_list() if not loss_df.is_empty() else []
    if steps:
        _plot_lines(
            steps,
            [
                loss_df["loss_total"].to_list(),
                loss_df["loss_distill"].to_list(),
                loss_df["loss_rank"].to_list(),
                loss_df["loss_struct"].to_list(),
            ],
            ["total", "distill", "rank", "struct"],
            "loss curves",
            output_path / "loss_curves.png",
        )
        _plot_lines(
            steps,
            [
                loss_df["student_norm_mean"].to_list(),
                loss_df["teacher_norm_mean"].to_list(),
                loss_df["student_norm_std"].to_list(),
                loss_df["teacher_norm_std"].to_list(),
            ],
            ["student_mean", "teacher_mean", "student_std", "teacher_std"],
            "norm stats",
            output_path / "norm_stats.png",
        )
        _plot_lines(
            steps,
            [
                loss_df["student_cos_mean"].to_list(),
                loss_df["teacher_cos_mean"].to_list(),
                loss_df["student_cos_std"].to_list(),
                loss_df["teacher_cos_std"].to_list(),
            ],
            ["student_mean", "teacher_mean", "student_std", "teacher_std"],
            "cosine stats",
            output_path / "cosine_stats.png",
        )
    eval_df = logs.filter(pl.col("vs_teacher_recall@10").is_not_null())
    if not eval_df.is_empty():
        _plot_lines(
            eval_df["step"].to_list(),
            [eval_df["vs_teacher_recall@10"].to_list()],
            ["recall@10"],
            "vs_teacher recall@10",
            output_path / "vs_teacher_recall_curve.png",
        )


def train_student(
    teacher_embeddings: np.ndarray,
    teacher_topk: np.ndarray,
    model_cfg: Dict,
    train_cfg: Dict,
    output_dir: str | Path,
    log_dir: str | Path,
) -> Tuple[nn.Module, Dict[str, object]]:
    config, eval_cfg = build_train_config(train_cfg)
    device = resolve_device(config.device)
    student_cfg = StudentConfig(
        model_type=model_cfg["type"],
        in_dim=model_cfg["in_dim"],
        out_dim=model_cfg["out_dim"],
        hidden_dim=model_cfg.get("hidden_dim"),
        dropout=model_cfg["dropout"],
        normalize=model_cfg["normalize"],
    )
    set_seed(config.seed)
    model = build_student(student_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = _build_lr_scheduler(
        optimizer, config.steps, config.warmup_steps, config.lr_schedule
    )
    autocast_context, scaler = _build_amp_context(config.amp, device)

    # Initialize dynamic batch sizer
    track_b_cfg = config.track_b
    num_positives = (
        config.multi_positive.num_positives if config.multi_positive.enabled else 1
    )
    num_negatives = 5  # Fixed in training loop

    batch_sizer = DynamicBatchSizer(
        config=config.dynamic_batch,
        device=device,
        embedding_dim=student_cfg.in_dim,
        output_dim=student_cfg.out_dim,
        num_positives=num_positives,
        num_negatives=num_negatives,
        track_b_enabled=track_b_cfg.enable,
        track_b_k_pos=track_b_cfg.k_pos,
        track_b_m_neg=track_b_cfg.m_neg,
        amp_mode=config.amp,
    )

    # Log VRAM info and batch size
    vram_info = get_vram_info(device)
    print(f"amp_mode={config.amp} device={device.type}")
    print(
        f"vram_info: total={vram_info.total / 1024**3:.1f}GB "
        f"free={vram_info.free / 1024**3:.1f}GB "
        f"used={vram_info.used / 1024**3:.1f}GB"
    )
    effective_batch_size = (
        batch_sizer.batch_size if config.dynamic_batch.enabled else config.batch_size
    )
    print(
        "train_effective_config "
        + f"effective_batch_size={effective_batch_size} "
        + f"dynamic_batch_enabled={config.dynamic_batch.enabled} "
        + f"steps={config.steps} lr={config.lr} weight_decay={config.weight_decay} "
        + f"lr_schedule={config.lr_schedule} warmup_steps={config.warmup_steps}"
    )
    teacher_tensor = torch.tensor(
        teacher_embeddings, dtype=torch.float32, device=device
    )
    teacher_topk_tensor = torch.tensor(teacher_topk, dtype=torch.long, device=device)
    num_items = int(teacher_tensor.shape[0])
    teacher_norm = _norm_stats(teacher_embeddings)
    rng = np.random.default_rng(0)
    teacher_cos = _cosine_stats(teacher_embeddings, 5000, rng)
    stats = {"loss": 0.0}
    log_rows: List[Dict[str, object]] = []
    step = 0
    if config.resume_path:
        resume_path = Path(config.resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_path not found: {resume_path}")
        step, has_scheduler_state = load_training_state(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        if not has_scheduler_state and step > 0:
            scheduler.last_epoch = step - 1
    progress = tqdm(total=config.steps, desc="train student", initial=step)
    best_recall = None
    best_step = None
    best_checkpoint = None
    warned_empty_hard = False
    positive_k = min(10, teacher_topk_tensor.shape[1])
    num_negatives = 5

    track_b_cfg = config.track_b
    queue: Optional[TeacherEmbeddingQueue] = None
    if track_b_cfg.enable:
        use_cpu_storage = resolve_queue_storage(
            prefer_device_storage=not track_b_cfg.queue_cpu_fallback,
            queue_size=track_b_cfg.queue_size,
            embedding_dim=int(teacher_tensor.shape[1]),
            device=device,
        )
        queue = TeacherEmbeddingQueue(
            embedding_dim=int(teacher_tensor.shape[1]),
            size=track_b_cfg.queue_size,
            device=device,
            use_cpu_storage=use_cpu_storage,
        )
    # num_positives already defined above for batch_sizer
    last_logged_batch_size = effective_batch_size
    try:
        while step < config.steps:
            # Get current batch size (dynamic or fixed)
            if config.dynamic_batch.enabled:
                batch_size = batch_sizer.batch_size
            else:
                batch_size = config.batch_size

            # OOM retry loop
            oom_retry_count = 0
            max_oom_retries = (
                config.dynamic_batch.max_oom_retries
                if config.dynamic_batch.oom_retry_enabled
                else 0
            )

            while True:
                try:
                    anchor_indices = torch.randint(0, num_items, (batch_size,), device=device)
                    positive_indices = _sample_positive_indices(
                        teacher_topk_tensor, anchor_indices, positive_k, num_positives
                    )
                    neighbor_indices = teacher_topk_tensor[anchor_indices, :positive_k]
                    teacher_anchor = teacher_tensor[anchor_indices]
                    teacher_positive = teacher_tensor[positive_indices]
                    teacher_neighbor = teacher_tensor[neighbor_indices]
                    teacher_neg = None
                    if queue is not None:
                        queue.enqueue(teacher_anchor)
                    if config.use_rank:
                        hard_cfg = config.hard_negative
                        hard_neg = None
                        if hard_cfg.enabled:
                            if hard_cfg.mode != "teacher_tail":
                                raise ValueError("Unsupported hard negative mode")
                            hard_neg = _sample_hard_negatives(
                                teacher_topk_tensor,
                                anchor_indices,
                                hard_cfg.tail_from,
                                hard_cfg.tail_to,
                                num_negatives,
                            )
                            if hard_neg is None and not warned_empty_hard:
                                print("hard negative pool empty; falling back to random")
                                warned_empty_hard = True
                        random_neg = _sample_random_negatives(
                            num_items,
                            anchor_indices,
                            positive_indices,
                            num_negatives,
                        )
                        neg_indices = _mix_negatives(
                            hard_neg, random_neg, hard_cfg.mix_random_ratio
                        )
                        neg_indices = _ensure_valid_negatives(
                            neg_indices,
                            num_items,
                            anchor_indices,
                            positive_indices,
                        )
                        teacher_neg = teacher_tensor[neg_indices]
                    optimizer.zero_grad()
                    with autocast_context:
                        student_anchor = model(teacher_anchor)
                        student_positive = model(
                            teacher_positive.view(-1, teacher_positive.shape[-1])
                        ).view(teacher_positive.shape[0], teacher_positive.shape[1], -1)
                        student_neighbor = model(
                            teacher_neighbor.view(-1, teacher_neighbor.shape[-1])
                        ).view(teacher_neighbor.shape[0], teacher_neighbor.shape[1], -1)
                        student_neg = None
                        if config.use_rank:
                            if teacher_neg is None:
                                raise RuntimeError(
                                    "rank loss enabled but teacher_neg was not sampled"
                                )
                            student_neg = model(
                                teacher_neg.view(-1, teacher_neg.shape[-1])
                            ).view(teacher_neg.shape[0], teacher_neg.shape[1], -1)
                        if config.use_distill:
                            distill = distill_loss(
                                teacher_anchor, student_anchor, config.distill_mode
                            )
                        else:
                            distill = torch.tensor(0.0, device=device)
                        rank = torch.tensor(0.0, device=device)
                        if config.use_rank:
                            if student_neg is None:
                                raise RuntimeError("rank loss enabled but student_neg is None")
                            if config.rank_kind != "info_nce":
                                raise ValueError("Unsupported rank kind")
                            student_neg_tensor = cast(torch.Tensor, student_neg)
                            if config.multi_positive.enabled:
                                rank = info_nce_multi(
                                    student_anchor,
                                    student_positive,
                                    student_neg_tensor,
                                    temperature=config.rank_temperature,
                                )
                            else:
                                rank = info_nce(
                                    student_anchor,
                                    student_positive[:, 0],
                                    student_neg_tensor,
                                    temperature=config.rank_temperature,
                                )
                        if config.use_struct:
                            struct = distortion_loss(
                                teacher_anchor,
                                student_anchor,
                                teacher_neighbor,
                                student_neighbor,
                            )
                        else:
                            struct = torch.tensor(0.0, device=device)
                        loss_track_a = mix_losses(
                            distill, rank, struct, config.loss_mix, step, config.steps
                        )

                        loss_listwise = torch.tensor(0.0, device=device)
                        p_t_entropy = torch.tensor(0.0, device=device)
                        filtered_neg_ratio = 0.0
                        if track_b_cfg.enable:
                            if queue is None:
                                raise RuntimeError("track_b enabled but queue is None")
                            if queue.num_filled <= 0:
                                warm_n = min(track_b_cfg.queue_size, num_items)
                                perm = torch.randperm(num_items, device=device)[:warm_n]
                                queue.enqueue(teacher_tensor[perm])

                            sampled = queue.sample(
                                track_b_cfg.m_neg * batch_size, device=device
                            )
                            queue_negs = reshape_queue_samples(
                                sampled, batch_size, track_b_cfg.m_neg
                            )

                            teacher_pos_vecs = teacher_tensor[
                                teacher_topk_tensor[anchor_indices, : track_b_cfg.k_pos]
                            ]

                            candidates, filtered_neg_ratio = build_candidate_vectors(
                                teacher_anchor,
                                teacher_pos_vecs,
                                queue_negs,
                                false_neg_filter_mode=track_b_cfg.false_neg_filter_mode,
                                false_neg_threshold=track_b_cfg.false_neg_threshold,
                                false_neg_top_percent=track_b_cfg.false_neg_top_percent,
                            )

                            teacher_scores = torch.einsum(
                                "bd,bnd->bn",
                                F.normalize(teacher_anchor, dim=-1),
                                F.normalize(candidates, dim=-1),
                            )

                            student_anchor_b = student_anchor
                            student_cand_b = model(
                                candidates.view(-1, candidates.shape[-1])
                            ).view(candidates.shape[0], candidates.shape[1], -1)
                            student_scores = torch.einsum(
                                "bd,bnd->bn",
                                F.normalize(student_anchor_b, dim=-1),
                                F.normalize(student_cand_b, dim=-1),
                            )

                            loss_listwise, p_t_entropy = listwise_kl_distill(
                                teacher_scores, student_scores, tau=track_b_cfg.tau
                            )

                        loss = (
                            1.0 - track_b_cfg.mix_lambda
                        ) * loss_track_a + track_b_cfg.mix_lambda * loss_listwise
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    # Step completed successfully
                    if config.dynamic_batch.enabled:
                        batch_sizer.on_step_success()

                    # Break out of OOM retry loop on success
                    break

                except Exception as e:
                    if is_oom_error(e) and config.dynamic_batch.oom_retry_enabled:
                        oom_retry_count += 1
                        if oom_retry_count > max_oom_retries:
                            raise RuntimeError(
                                f"Max OOM retries ({max_oom_retries}) exceeded. "
                                f"Last batch size: {batch_size}. "
                                "Try reducing batch_size, using amp=fp16, or a smaller model."
                            ) from e

                        # Clear memory and reduce batch size
                        clear_memory_cache(device)
                        if config.dynamic_batch.enabled:
                            batch_size = batch_sizer.on_oom()
                        else:
                            # For fixed batch mode, apply manual reduction
                            batch_size = max(
                                config.dynamic_batch.min_batch_size,
                                int(batch_size * config.dynamic_batch.oom_reduction_factor),
                            )

                        print(
                            f"OOM at step {step}, retry {oom_retry_count}/{max_oom_retries}, "
                            f"reducing batch_size to {batch_size}"
                        )
                        # Re-sample with smaller batch
                        continue
                    else:
                        # Non-OOM error, re-raise
                        raise

            scheduler.step()
            stats["loss"] = float(loss.detach().cpu().item())
            step += 1
            progress.update(1)

            # Log batch size changes
            current_bs = batch_sizer.batch_size if config.dynamic_batch.enabled else batch_size
            if current_bs != last_logged_batch_size:
                print(f"Batch size changed: {last_logged_batch_size} -> {current_bs} at step {step}")
                last_logged_batch_size = current_bs
            if (
                config.save_every > 0 and step % config.save_every == 0
            ) or step == config.steps:
                save_checkpoint(
                    model,
                    output_dir,
                    step,
                    student_cfg,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                )
            log_step = step % config.log_every == 0 or step == config.steps
            eval_step = (
                eval_cfg.eval_every > 0 and step % eval_cfg.eval_every == 0
            ) or step == config.steps
            if log_step or eval_step:
                student_norm = _norm_stats_torch(student_anchor)
                sample_size = min(2000, int(student_anchor.shape[0]) * 2)
                student_cos = _cosine_stats_torch(student_anchor, sample_size)
                collapse_cos = student_cos["std"] < teacher_cos["std"] * 0.33
                collapse_norm = (
                    student_norm["std"] < teacher_norm["std"] * 0.33
                    or student_norm["std"] > teacher_norm["std"] * 3.0
                )
                eval_metrics: Dict[str, Any] = {
                    "eval_status": None,
                    "vs_teacher_recall@10": None,
                    "vs_teacher_ndcg@10": None,
                    "vs_teacher_overlap@10": None,
                    "vs_student_recall@10": None,
                }
                if eval_step:
                    eval_metrics = _evaluate(
                        model,
                        teacher_embeddings,
                        teacher_topk,
                        eval_cfg,
                        device,
                        batch_size=None,
                        output_dim=student_cfg.out_dim,
                    )
                    recall_value = eval_metrics.get("vs_teacher_recall@10")
                    recall = (
                        float(recall_value)
                        if isinstance(recall_value, (int, float))
                        else None
                    )
                    if recall is not None:
                        if best_recall is None or recall > best_recall:
                            best_recall = recall
                            best_step = step
                            best_checkpoint = save_named_checkpoint(
                                model, output_dir, "student_best.pt", student_cfg
                            )
                # Get batch size stats for logging
                batch_stats = batch_sizer.get_stats() if config.dynamic_batch.enabled else {}
                log_rows.append(
                    {
                        "step": step,
                        "batch_size": batch_size,
                        "dynamic_batch_enabled": config.dynamic_batch.enabled,
                        "batch_oom_count": batch_stats.get("oom_count", 0),
                        "loss_total": float(loss.detach().cpu().item()),
                        "loss_distill": float(distill.detach().cpu().item()),
                        "loss_rank": float(rank.detach().cpu().item()),
                        "loss_struct": float(struct.detach().cpu().item()),
                        "loss_track_a": float(loss_track_a.detach().cpu().item()),
                        "loss_listwise": float(loss_listwise.detach().cpu().item()),
                        "teacher_norm_mean": teacher_norm["mean"],
                        "teacher_norm_std": teacher_norm["std"],
                        "teacher_norm_p50": teacher_norm["p50"],
                        "teacher_norm_p95": teacher_norm["p95"],
                        "student_norm_mean": student_norm["mean"],
                        "student_norm_std": student_norm["std"],
                        "student_norm_p50": student_norm["p50"],
                        "student_norm_p95": student_norm["p95"],
                        "teacher_cos_mean": teacher_cos["mean"],
                        "teacher_cos_std": teacher_cos["std"],
                        "teacher_cos_p5": teacher_cos["p5"],
                        "teacher_cos_p95": teacher_cos["p95"],
                        "student_cos_mean": student_cos["mean"],
                        "student_cos_std": student_cos["std"],
                        "student_cos_p5": student_cos["p5"],
                        "student_cos_p95": student_cos["p95"],
                        "collapse_suspect": bool(collapse_cos or collapse_norm),
                        "num_positives": num_positives,
                        "mix_random_ratio": config.hard_negative.mix_random_ratio,
                        "tail_to": config.hard_negative.tail_to,
                        "track_b_enable": bool(track_b_cfg.enable),
                        "track_b_tau": float(track_b_cfg.tau),
                        "track_b_k": int(track_b_cfg.k_pos),
                        "track_b_m": int(track_b_cfg.m_neg),
                        "track_b_queue_size": int(track_b_cfg.queue_size),
                        "track_b_false_neg_filter_mode": str(
                            track_b_cfg.false_neg_filter_mode
                        ),
                        "track_b_false_neg_threshold": float(
                            track_b_cfg.false_neg_threshold
                        ),
                        "track_b_false_neg_top_percent": float(
                            track_b_cfg.false_neg_top_percent
                        ),
                        "track_b_mix_lambda": float(track_b_cfg.mix_lambda),
                        "track_b_p_t_entropy": float(p_t_entropy.detach().cpu().item()),
                        "track_b_filtered_neg_ratio": float(filtered_neg_ratio),
                        "eval_status": eval_metrics.get("eval_status"),
                        "vs_teacher_recall@10": eval_metrics.get(
                            "vs_teacher_recall@10"
                        ),
                        "vs_teacher_ndcg@10": eval_metrics.get("vs_teacher_ndcg@10"),
                        "vs_teacher_overlap@10": eval_metrics.get(
                            "vs_teacher_overlap@10"
                        ),
                        "vs_student_recall@10": eval_metrics.get(
                            "vs_student_recall@10"
                        ),
                    }
                )
    finally:
        progress.close()
    log_path = ensure_dir(Path(log_dir))
    jsonl_path = log_path / "train_logs.jsonl"
    _write_log_rows_jsonl(log_rows, jsonl_path)

    normalized_rows, diag, schema_overrides = _normalize_log_rows_for_polars(log_rows)
    if diag.mixed_type_columns:
        parts: List[str] = []
        for k, cnt in sorted(diag.mixed_type_columns.items()):
            types_s = ",".join(
                f"{t.__name__}:{n}"
                for t, n in sorted(cnt.items(), key=lambda x: x[0].__name__)
            )
            parts.append(f"{k}={{ {types_s} }}")
        print(
            "log schema normalization: mixed types detected; coerced to str: "
            + "; ".join(parts[:12])
            + ("; ..." if len(parts) > 12 else "")
        )

    from polars._typing import SchemaDict  # pyright: ignore[reportMissingImports]

    schema_overrides = cast(SchemaDict, schema_overrides)

    logs: Optional[pl.DataFrame] = None
    try:
        logs = pl.from_dicts(
            normalized_rows,
            schema_overrides=schema_overrides or None,
            strict=False,
            infer_schema_length=None,
        )
        logs.write_parquet(log_path / "train_logs.parquet")
        _make_train_plots(logs, log_path / "figures")
    except Exception as e:
        print(
            "warning: failed to build train_logs.parquet via Polars; "
            f"wrote JSONL instead at {jsonl_path}. error={type(e).__name__}: {e}"
        )
    summary = {
        "final_loss": stats["loss"],
        "steps": config.steps,
        "best_vs_teacher_recall@10": best_recall,
        "best_step": best_step,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "hard_negative": {
            "enabled": config.hard_negative.enabled,
            "mode": config.hard_negative.mode,
            "tail_from": config.hard_negative.tail_from,
            "tail_to": config.hard_negative.tail_to,
            "mix_random_ratio": config.hard_negative.mix_random_ratio,
        },
        "multi_positive": {
            "enabled": config.multi_positive.enabled,
            "num_positives": config.multi_positive.num_positives,
        },
    }
    save_json(log_path / "train_summary.json", summary)
    return model, summary


def save_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    step: int,
    config: StudentConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Path:
    output_path = ensure_dir(Path(output_dir))
    checkpoint_path = output_path / f"student_step_{step}.pt"
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "type": config.model_type,
            "in_dim": config.in_dim,
            "out_dim": config.out_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "normalize": config.normalize,
        },
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, checkpoint_path)
    save_json(
        output_path / "last_checkpoint.json",
        {"path": str(checkpoint_path), "out_dim": config.out_dim},
    )
    return checkpoint_path


def save_named_checkpoint(
    model: nn.Module, output_dir: str | Path, name: str, config: StudentConfig
) -> Path:
    output_path = ensure_dir(Path(output_dir))
    checkpoint_path = output_path / name
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "type": config.model_type,
            "in_dim": config.in_dim,
            "out_dim": config.out_dim,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
            "normalize": config.normalize,
        },
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path) -> Tuple[nn.Module, StudentConfig]:
    payload = torch.load(path, map_location="cpu")
    cfg = payload["config"]
    config = StudentConfig(
        model_type=cfg["type"],
        in_dim=cfg["in_dim"],
        out_dim=cfg["out_dim"],
        hidden_dim=cfg.get("hidden_dim"),
        dropout=cfg["dropout"],
        normalize=cfg["normalize"],
    )
    model = build_student(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config
