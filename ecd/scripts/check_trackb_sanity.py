#!/usr/bin/env python3
"""
Quick sanity check for Track-B v1 implementation.
Tests configuration loading, model initialization, and core functions.
"""

from __future__ import annotations

import torch

from ecd.models.student import StudentConfig, build_student
from ecd.train.track_b import (
    TrackBConfig,
    build_track_b_config,
    cosine_scores,
    listwise_kl_distill,
    false_negative_filter,
)
from ecd.train.memory_queue import TeacherEmbeddingQueue


def test_trackb_model():
    """Test Track-B projection head initialization."""
    cfg = StudentConfig(
        model_type="track_b",
        in_dim=768,
        out_dim=128,
        hidden_dim=None,
        dropout=0.0,
        normalize=True,
        track_b_use_skip=True,
        track_b_alpha_init=0.1,
    )

    model = build_student(cfg)

    x = torch.randn(4, 768)
    y = model(x)

    assert y.shape == (4, 128), f"Expected (4, 128), got {y.shape}"
    assert y.norm(dim=-1).allclose(torch.ones(4)), "Output should be L2-normalized"

    print("✓ Track-B model initialization: OK")
    return True


def test_trackb_config():
    """Test Track-B config building."""
    cfg = {
        "track_b": {
            "enable": True,
            "k_pos": 50,
            "m_neg": 1024,
            "tau": 0.07,
            "queue_size": 32000,
            "queue_cpu_fallback": True,
            "false_neg_filter": {
                "mode": "threshold",
                "threshold": 0.8,
                "top_percent": 0.02,
            },
            "mix": {"lambda": 1.0},
        }
    }

    tb_cfg = build_track_b_config(cfg)

    assert isinstance(tb_cfg, TrackBConfig)
    assert tb_cfg.enable is True
    assert tb_cfg.k_pos == 50
    assert tb_cfg.m_neg == 1024
    assert tb_cfg.tau == 0.07
    assert tb_cfg.false_neg_filter_mode == "threshold"
    assert tb_cfg.false_neg_threshold == 0.8
    assert tb_cfg.mix_lambda == 1.0

    print("✓ Track-B config building: OK")
    return True


def test_cosine_scores():
    """Test cosine similarity scoring."""
    a = torch.randn(2, 128)
    b = torch.randn(2, 5, 128)

    scores = cosine_scores(a, b)

    assert scores.shape == (2, 5)
    assert scores.min() >= -1.0 and scores.max() <= 1.0

    print("✓ Cosine scoring: OK")
    return True


def test_listwise_kl_distill():
    """Test listwise KL distillation loss."""
    teacher_scores = torch.randn(4, 100)
    student_scores = teacher_scores + 0.1 * torch.randn(4, 100)

    loss, entropy = listwise_kl_distill(teacher_scores, student_scores, tau=0.07)

    assert loss.item() >= 0.0
    assert entropy.item() >= 0.0
    assert torch.isfinite(loss)

    loss_z, _ = listwise_kl_distill(teacher_scores, teacher_scores, tau=0.07)
    assert loss_z.item() < 1e-6, "Loss should be ~0 when scores are identical"

    print("✓ Listwise KL distill: OK")
    return True


def test_false_negative_filter():
    """Test false negative filtering."""
    anchor = torch.randn(2, 128)
    neg = torch.randn(2, 20, 128)

    kept, ratio = false_negative_filter(
        anchor, neg, mode="none", threshold=0.8, top_percent=0.02
    )
    assert kept.shape == neg.shape
    assert ratio == 0.0

    kept_thresh, ratio_thresh = false_negative_filter(
        anchor, neg, mode="threshold", threshold=0.8, top_percent=0.02
    )
    assert kept_thresh.shape[0] == neg.shape[0]
    assert kept_thresh.shape[1] <= neg.shape[1]
    assert 0.0 <= ratio_thresh <= 1.0

    print("✓ False negative filtering: OK")
    return True


def test_memory_queue():
    """Test teacher embedding queue."""
    device = torch.device("cpu")
    queue = TeacherEmbeddingQueue(
        embedding_dim=128, size=100, device=device, use_cpu_storage=True
    )

    assert queue.num_filled == 0

    x1 = torch.randn(10, 128)
    queue.enqueue(x1)
    assert queue.num_filled == 10

    x2 = torch.randn(90, 128)
    queue.enqueue(x2)
    assert queue.num_filled == 100

    sampled = queue.sample(20, device=device)
    assert sampled.shape == (20, 128)

    x3 = torch.randn(15, 128)
    queue.enqueue(x3)
    assert queue.num_filled == 100

    print("✓ Memory queue: OK")
    return True


def main() -> int:
    """Run all sanity checks."""
    print("=" * 50)
    print("Track-B v1 Implementation Sanity Check")
    print("=" * 50)
    print()

    tests = [
        test_trackb_config,
        test_trackb_model,
        test_cosine_scores,
        test_listwise_kl_distill,
        test_false_negative_filter,
        test_memory_queue,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
