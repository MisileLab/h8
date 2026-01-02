import mlx.core as mx

from retnet_mlx.norms import TokenwiseGroupNorm


def test_tokenwise_groupnorm_matches_flattened():
    mx.random.seed(0)
    b, t, c = 2, 4, 8
    norm = TokenwiseGroupNorm(num_groups=2, dims=c)
    x = mx.random.normal((b, t, c))
    y = norm(x)
    y_flat = norm.norm(x.reshape(b * t, c)).reshape(b, t, c)
    assert mx.allclose(y, y_flat, rtol=1e-5, atol=1e-5).item()
