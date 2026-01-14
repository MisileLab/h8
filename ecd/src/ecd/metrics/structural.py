from __future__ import annotations

from typing import Dict

import numpy as np


def distortion(
    teacher_vectors: np.ndarray, student_vectors: np.ndarray, pairs: np.ndarray
) -> float:
    teacher = teacher_vectors[pairs[:, 0]]
    teacher_neighbor = teacher_vectors[pairs[:, 1]]
    student = student_vectors[pairs[:, 0]]
    student_neighbor = student_vectors[pairs[:, 1]]
    teacher_sim = np.sum(teacher * teacher_neighbor, axis=1)
    student_sim = np.sum(student * student_neighbor, axis=1)
    return float(np.mean((teacher_sim - student_sim) ** 2))


def rank_correlation(teacher_scores: np.ndarray, student_scores: np.ndarray) -> float:
    teacher_rank = np.argsort(np.argsort(-teacher_scores))
    student_rank = np.argsort(np.argsort(-student_scores))
    teacher_rank = teacher_rank.astype(np.float32)
    student_rank = student_rank.astype(np.float32)
    teacher_rank -= teacher_rank.mean()
    student_rank -= student_rank.mean()
    denom = np.linalg.norm(teacher_rank) * np.linalg.norm(student_rank)
    if denom == 0:
        return 0.0
    return float(np.dot(teacher_rank, student_rank) / denom)


def compute_structural_metrics(
    teacher_scores: np.ndarray, student_scores: np.ndarray, pairs: np.ndarray
) -> Dict[str, float]:
    return {
        "distortion": distortion(teacher_scores, student_scores, pairs),
        "rank_corr": rank_correlation(teacher_scores, student_scores),
    }
