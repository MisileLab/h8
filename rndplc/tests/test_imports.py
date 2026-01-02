def test_imports():
    import retnet_mlx
    from retnet_mlx import norms, ops

    assert retnet_mlx.__version__
    assert hasattr(norms, "TokenwiseGroupNorm")
    assert hasattr(norms, "RMSNorm")
    assert hasattr(ops, "DecayMaskCache")
