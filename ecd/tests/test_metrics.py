import numpy as np

from ecd.metrics.retrieval import mrr_at_k, ndcg_at_k, recall_at_k


def test_metrics_basic():
    truth = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    pred = np.array([[1, 3, 9], [6, 5, 4]], dtype=np.int32)
    assert recall_at_k(pred, truth, 3) == 0.8333333333333333
    assert mrr_at_k(pred, truth, 3) >= 0.5
    assert ndcg_at_k(pred, truth, 3) > 0.5
