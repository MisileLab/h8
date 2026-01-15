from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrackBConfig:
    enable: bool

    k_pos: int
    m_neg: int
    tau: float

    queue_size: int
    queue_cpu_fallback: bool

    false_neg_filter_mode: Literal["none", "threshold", "top_percent"]
    false_neg_threshold: float
    false_neg_top_percent: float

    mix_lambda: float


def build_track_b_config(cfg: dict) -> TrackBConfig:
    tb = cfg.get("track_b", {})
    fn = tb.get("false_neg_filter", {})
    mix = tb.get("mix", {})

    mode = str(fn.get("mode", "threshold"))
    if mode not in {"none", "threshold", "top_percent"}:
        raise ValueError(f"Unsupported false_neg_filter.mode={mode}")

    return TrackBConfig(
        enable=bool(tb.get("enable", False)),
        k_pos=int(tb.get("k_pos", 50)),
        m_neg=int(tb.get("m_neg", 1024)),
        tau=float(tb.get("tau", 0.07)),
        queue_size=int(tb.get("queue_size", 32000)),
        queue_cpu_fallback=bool(tb.get("queue_cpu_fallback", True)),
        false_neg_filter_mode=mode,  # type: ignore[assignment]
        false_neg_threshold=float(fn.get("threshold", 0.8)),
        false_neg_top_percent=float(fn.get("top_percent", 0.02)),
        mix_lambda=float(mix.get("lambda", 1.0)),
    )


def cosine_scores(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return torch.einsum("bd,bnd->bn", a, b)


@torch.no_grad()
def false_negative_filter(
    teacher_anchor: torch.Tensor,
    neg_vectors: torch.Tensor,
    mode: Literal["none", "threshold", "top_percent"],
    threshold: float,
    top_percent: float,
) -> Tuple[torch.Tensor, float]:
    if mode == "none":
        return neg_vectors, 0.0

    teacher_anchor = F.normalize(teacher_anchor, dim=-1)
    neg_vectors = F.normalize(neg_vectors, dim=-1)

    # scores: (B, N)
    scores = torch.einsum("bd,bnd->bn", teacher_anchor, neg_vectors)

    if mode == "threshold":
        keep = scores <= float(threshold)
    else:
        # mode == top_percent
        n = int(scores.shape[1])
        if n <= 1:
            return neg_vectors, 0.0
        k_drop = max(0, min(n - 1, int(round(float(top_percent) * n))))
        if k_drop <= 0:
            return neg_vectors, 0.0
        # drop the top k_drop per row.
        _, idx = torch.topk(scores, k=k_drop, dim=1, largest=True, sorted=False)
        keep = torch.ones_like(scores, dtype=torch.bool)
        keep.scatter_(1, idx, False)

    kept_list = []
    for row in range(int(neg_vectors.shape[0])):
        kept_list.append(neg_vectors[row, keep[row]])

    min_kept = min(int(x.shape[0]) for x in kept_list)
    if min_kept <= 0:
        scores_row = scores
        _, drop_idx = torch.topk(scores_row, k=1, dim=1, largest=True, sorted=False)
        keep_min = torch.ones_like(scores_row, dtype=torch.bool)
        keep_min.scatter_(1, drop_idx, False)
        kept_list = [
            neg_vectors[row, keep_min[row]] for row in range(int(neg_vectors.shape[0]))
        ]
        min_kept = min(int(x.shape[0]) for x in kept_list)

    kept = torch.stack([x[:min_kept] for x in kept_list], dim=0)

    filtered_ratio = 1.0 - float(keep.float().mean().item())
    return kept, filtered_ratio


def listwise_kl_distill(
    teacher_scores: torch.Tensor,
    student_scores: torch.Tensor,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if teacher_scores.shape != student_scores.shape:
        raise ValueError("teacher_scores and student_scores must have same shape")

    t = float(tau)
    if t <= 0:
        raise ValueError("tau must be > 0")

    log_p_t = F.log_softmax(teacher_scores / t, dim=1)
    log_p_s = F.log_softmax(student_scores / t, dim=1)
    p_t = log_p_t.exp()

    # KL(p_t || p_s) = sum p_t * (log_p_t - log_p_s)
    loss = torch.sum(p_t * (log_p_t - log_p_s), dim=1).mean()

    entropy = -torch.sum(p_t * log_p_t, dim=1).mean()
    return loss, entropy


def build_candidate_vectors(
    teacher_anchor: torch.Tensor,
    teacher_topk_vectors: torch.Tensor,
    queue_neg_vectors: torch.Tensor,
    false_neg_filter_mode: Literal["none", "threshold", "top_percent"],
    false_neg_threshold: float,
    false_neg_top_percent: float,
) -> Tuple[torch.Tensor, float]:
    neg_kept, filtered_ratio = false_negative_filter(
        teacher_anchor,
        queue_neg_vectors,
        mode=false_neg_filter_mode,
        threshold=false_neg_threshold,
        top_percent=false_neg_top_percent,
    )

    candidates = torch.cat([teacher_topk_vectors, neg_kept], dim=1)
    return candidates, float(filtered_ratio)


def candidates_from_topk_and_queue(
    teacher_embeddings: torch.Tensor,
    teacher_topk: torch.Tensor,
    anchor_indices: torch.Tensor,
    k_pos: int,
    queue_samples: torch.Tensor,
) -> torch.Tensor:
    topk_idx = teacher_topk[anchor_indices, :k_pos]
    pos = teacher_embeddings[topk_idx]
    # queue_samples is already (B, M, D)
    return torch.cat([pos, queue_samples], dim=1)


def reshape_queue_samples(
    sampled: torch.Tensor, batch_size: int, m_neg: int
) -> torch.Tensor:
    if sampled.shape[0] != batch_size * m_neg:
        raise ValueError("sampled must be flat with batch_size*m_neg rows")
    return sampled.view(batch_size, m_neg, -1)
