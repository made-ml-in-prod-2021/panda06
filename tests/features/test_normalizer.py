import numpy as np

from ml_project.features import Normalizer

EPS = 1e-8


def test_normalizer():
    normalizer = Normalizer()
    assert normalizer.mean is None and normalizer.std is None
    x = np.random.rand(5, 2)
    x = normalizer.fit_transform(x)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    assert np.all((-EPS <= means) & (means <= EPS))
    assert np.all((1 - EPS <= stds) & (stds <= 1 + EPS))
