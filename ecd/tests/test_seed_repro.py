import numpy as np

from ecd.utils.seed import set_seed


def test_seed_reproducibility():
    set_seed(123)
    first = np.random.rand(4)
    set_seed(123)
    second = np.random.rand(4)
    assert np.allclose(first, second)
