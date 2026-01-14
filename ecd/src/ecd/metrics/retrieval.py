from __future__ import annotations

from typing import Dict, List

import numpy as np


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    pred_k = pred[:, :k]
    truth_k = truth[:, :k]
    hits = [
        len(set(pred_k[i]).intersection(set(truth_k[i])))
        for i in range(pred_k.shape[0])
    ]
    return float(np.mean(np.array(hits) / k))


def overlap_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    return recall_at_k(pred, truth, k)


def ndcg_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    pred_k = pred[:, :k]
    truth_k = truth[:, :k]
    scores = []
    for i in range(pred_k.shape[0]):
        relevance = {doc_id: 1.0 for doc_id in truth_k[i]}
        gains = [relevance.get(doc_id, 0.0) for doc_id in pred_k[i]]
        dcg = np.sum([gain / np.log2(idx + 2) for idx, gain in enumerate(gains)])
        ideal = np.sum([1.0 / np.log2(idx + 2) for idx in range(len(truth_k[i]))])
        scores.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(scores))


def mrr_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    pred_k = pred[:, :k]
    truth_k = truth[:, :k]
    scores = []
    for i in range(pred_k.shape[0]):
        rank = 0
        truth_set = set(truth_k[i])
        for idx, doc_id in enumerate(pred_k[i]):
            if doc_id in truth_set:
                rank = idx + 1
                break
        scores.append(1.0 / rank if rank > 0 else 0.0)
    return float(np.mean(scores))


def compute_metrics(
    pred: np.ndarray, truth: np.ndarray, k_values: List[int]
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for k in k_values:
        output[f"recall@{k}"] = recall_at_k(pred, truth, k)
        output[f"overlap@{k}"] = overlap_at_k(pred, truth, k)
        output[f"ndcg@{k}"] = ndcg_at_k(pred, truth, k)
        output[f"mrr@{k}"] = mrr_at_k(pred, truth, k)
    return output
