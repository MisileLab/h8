import torch

from ecd.train.memory_queue import TeacherEmbeddingQueue
from ecd.train.track_b import false_negative_filter, listwise_kl_distill


def test_teacher_embedding_queue_fifo_shapes() -> None:
    device = torch.device("cpu")
    q = TeacherEmbeddingQueue(
        embedding_dim=4, size=8, device=device, use_cpu_storage=True
    )

    x1 = torch.arange(0, 12, dtype=torch.float32).view(3, 4)
    q.enqueue(x1)
    assert q.num_filled == 3

    x2 = torch.arange(100, 120, dtype=torch.float32).view(5, 4)
    q.enqueue(x2)
    assert q.num_filled == 8

    # overwrite (FIFO)
    x3 = torch.arange(200, 216, dtype=torch.float32).view(4, 4)
    q.enqueue(x3)
    assert q.num_filled == 8

    s = q.sample(6, device=device)
    assert s.shape == (6, 4)


def test_listwise_kl_distill_zero_when_scores_equal() -> None:
    teacher = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.25, 0.0]])
    loss, ent = listwise_kl_distill(teacher, teacher.clone(), tau=0.07)
    assert torch.isfinite(loss)
    assert torch.isfinite(ent)
    assert float(loss.item()) < 1e-6


def test_false_negative_filter_threshold() -> None:
    # anchor aligned with first dimension
    anchor = torch.tensor([[1.0, 0.0, 0.0]])

    # one true-like (cos=1), one borderline (cos~0.707), one far (cos=0)
    negs = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    kept, ratio = false_negative_filter(
        teacher_anchor=anchor,
        neg_vectors=negs,
        mode="threshold",
        threshold=0.8,
        top_percent=0.02,
    )
    assert kept.shape[0] == 1
    assert kept.shape[2] == 3
    assert 0.0 <= ratio <= 1.0
    # should drop the first (cos=1)
    assert kept.shape[1] == 2
