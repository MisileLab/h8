import mlx.core as mx

from retnet_mlx.config import DecayConfig
from retnet_mlx.decay_policy import LearnedStaticDecayPolicy


def test_learned_static_monotonic_heads():
    mx.random.seed(0)
    config = DecayConfig(
        policy="learned_static",
        gamma_min=0.1,
        gamma_max=0.999,
        monotonic_heads=True,
        h_min=1.0,
        min_head_gap=0.1,
    )
    policy = LearnedStaticDecayPolicy(config, n_heads=6, d_model=32)
    gamma = policy.gamma()

    assert mx.all(gamma[1:] > gamma[:-1]).item()
    assert (mx.min(gamma) >= config.gamma_min).item()
    assert (mx.max(gamma) <= config.gamma_max).item()

    x = mx.random.normal((2, 4, 32))
    alpha = policy.alpha(x, "parallel", layer_idx=0)
    assert mx.all(mx.isfinite(alpha)).item()
