import mlx.core as mx

from retnet_mlx.config import DecayConfig
from retnet_mlx.decay_policy import FixedDecayPolicy, LearnedStaticDecayPolicy
from retnet_mlx.ops import build_decay_mask_from_gamma
from retnet_mlx.retention import (
    chunkwise_retention,
    parallel_retention_fast_fixed,
    recurrent_retention_step,
)


def _rollout_recurrent(q, k, v, alpha):
    b, h, t, d = q.shape
    dv = v.shape[-1]
    past = mx.zeros((b, h, d, dv), dtype=mx.float32)
    outs = []
    for idx in range(t):
        alpha_t = alpha if alpha.ndim <= 1 else alpha[:, :, idx]
        out_t, past = recurrent_retention_step(
            q[:, :, idx], k[:, :, idx], v[:, :, idx], past, alpha_t
        )
        outs.append(out_t)
    return mx.stack(outs, axis=2)


def _rollout_chunkwise(q, k, v, alpha, chunk_size):
    b, h, t, _ = q.shape
    past = None
    outs = []
    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        q_chunk = q[:, :, start:end]
        k_chunk = k[:, :, start:end]
        v_chunk = v[:, :, start:end]
        if alpha.ndim <= 1:
            alpha_chunk = mx.broadcast_to(alpha[None, :, None], (b, h, end - start))
        else:
            alpha_chunk = alpha[:, :, start:end]
        out_chunk, past = chunkwise_retention(
            q_chunk, k_chunk, v_chunk, past, alpha_chunk
        )
        outs.append(out_chunk)
    return mx.concatenate(outs, axis=2)


def _run_equivalence(policy):
    mx.random.seed(0)
    b, h, t, dqk, dv = 2, 3, 16, 8, 16
    q = mx.random.normal((b, h, t, dqk))
    k = mx.random.normal((b, h, t, dqk))
    v = mx.random.normal((b, h, t, dv))

    alpha = policy.alpha(mx.zeros((b, t, 1)), "parallel", layer_idx=0)
    decay_mask = build_decay_mask_from_gamma(alpha, t)

    out_parallel = parallel_retention_fast_fixed(q, k, v, decay_mask)
    out_recurrent = _rollout_recurrent(q, k, v, alpha)

    assert mx.allclose(out_parallel, out_recurrent, rtol=1e-4, atol=1e-4).item()

    for chunk_size in (4, 8):
        out_chunk = _rollout_chunkwise(q, k, v, alpha, chunk_size)
        assert mx.allclose(out_parallel, out_chunk, rtol=1e-4, atol=1e-4).item()


def test_retention_equivalence_fixed():
    config = DecayConfig(policy="fixed", gamma_min=0.1, gamma_max=0.999)
    policy = FixedDecayPolicy(config, n_heads=3, d_model=32)
    _run_equivalence(policy)


def test_retention_equivalence_learned_static():
    config = DecayConfig(
        policy="learned_static", gamma_min=0.1, gamma_max=0.999, monotonic_heads=True
    )
    policy = LearnedStaticDecayPolicy(config, n_heads=3, d_model=32)
    _run_equivalence(policy)
