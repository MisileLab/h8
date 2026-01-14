from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def sample_hard_negatives(
    teacher_topk: np.ndarray,
    rank_start: int,
    rank_end: int,
    num_samples: int,
) -> np.ndarray:
    pool = teacher_topk[:, rank_start:rank_end]
    if pool.shape[1] == 0:
        raise ValueError("Hard negative pool is empty")
    choices = np.random.randint(0, pool.shape[1], size=(pool.shape[0], num_samples))
    return np.take_along_axis(pool, choices, axis=1)


def info_nce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    anchors = torch.nn.functional.normalize(anchors, dim=-1)
    positives = torch.nn.functional.normalize(positives, dim=-1)
    negatives = torch.nn.functional.normalize(negatives, dim=-1)
    pos_logits = torch.sum(anchors * positives, dim=-1, keepdim=True)
    neg_logits = torch.einsum("bd,bnd->bn", anchors, negatives)
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def info_nce_multi(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    anchors = torch.nn.functional.normalize(anchors, dim=-1)
    positives = torch.nn.functional.normalize(positives, dim=-1)
    negatives = torch.nn.functional.normalize(negatives, dim=-1)
    pos_logits = torch.einsum("bd,bpd->bp", anchors, positives)
    pos_logit = torch.logsumexp(pos_logits, dim=1, keepdim=True)
    neg_logits = torch.einsum("bd,bnd->bn", anchors, negatives)
    logits = torch.cat([pos_logit, neg_logits], dim=1) / temperature
    labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def build_negatives(
    student_vectors: torch.Tensor,
    negative_indices: np.ndarray,
) -> torch.Tensor:
    flat = student_vectors[negative_indices.reshape(-1)]
    return flat.view(negative_indices.shape[0], negative_indices.shape[1], -1)


def mix_negatives(
    hard_negative_indices: np.ndarray,
    random_negative_indices: np.ndarray,
    mix_ratio: float,
) -> np.ndarray:
    if mix_ratio <= 0.0:
        return random_negative_indices
    if mix_ratio >= 1.0:
        return hard_negative_indices
    mix_mask = np.random.rand(*hard_negative_indices.shape) < mix_ratio
    mixed = np.where(mix_mask, hard_negative_indices, random_negative_indices)
    return mixed
