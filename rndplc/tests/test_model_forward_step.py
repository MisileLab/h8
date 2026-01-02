import mlx.core as mx

from retnet_mlx.config import ModelConfig
from retnet_mlx.model import RetNetLM


def test_model_forward_step_matches_parallel():
    mx.random.seed(0)
    config = ModelConfig(
        vocab_size=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_qk=8,
        d_v=16,
        ffn_mult=2,
    )
    model = RetNetLM(config)

    b, t = 2, 6
    tokens = mx.random.randint(0, config.vocab_size, shape=(b, t))
    logits_full = model(tokens, mode="parallel")

    cache = None
    step_logits = []
    for idx in range(t):
        logits_t, cache = model.forward_step(tokens[:, idx], cache)
        step_logits.append(logits_t)
    logits_step = mx.stack(step_logits, axis=1)

    assert mx.allclose(logits_full, logits_step, rtol=1e-4, atol=1e-4).item()
