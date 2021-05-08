import numpy as np
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError


class Normalizer(TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise NotFittedError("Normalizer instance is not fitted yet")
        return (X - self.mean) / self.std

    def fit(self, X: np.ndarray) -> "Normalizer":
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
