import mlx.core as mx

from retnet_mlx.retention import parallel_retention_general, recurrent_retention_step


def _rollout_recurrent(q, k, v, alpha):
    b, h, t, d = q.shape
    dv = v.shape[-1]
    past = mx.zeros((b, h, d, dv), dtype=mx.float32)
    outs = []
    for idx in range(t):
        out_t, past = recurrent_retention_step(
            q[:, :, idx], k[:, :, idx], v[:, :, idx], past, alpha[:, :, idx]
        )
        outs.append(out_t)
    return mx.stack(outs, axis=2)


def test_general_alpha_equivalence_smallT():
    mx.random.seed(0)
    b, h, t, dqk, dv = 2, 3, 16, 8, 16
    q = mx.random.normal((b, h, t, dqk))
    k = mx.random.normal((b, h, t, dqk))
    v = mx.random.normal((b, h, t, dv))
    alpha = mx.random.uniform(low=0.05, high=0.95, shape=(b, h, t))

    out_parallel = parallel_retention_general(q, k, v, alpha)
    out_recurrent = _rollout_recurrent(q, k, v, alpha)

    assert mx.allclose(out_parallel, out_recurrent, rtol=1e-4, atol=1e-4).item()
