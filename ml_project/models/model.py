import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]


class Model:
    def __init__(self, model: SklearnRegressionModel):
        self.model = model

    def train_model(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    @staticmethod
    def evaluate_model(predicts: np.ndarray, targets: pd.Series) -> Dict[str, float]:
        return {
            "rmse": mean_squared_error(targets, predicts, squared=False),
            "mae": mean_absolute_error(targets, predicts),
        }

    def serialize(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize(path: str) -> "Model":
        with open(path, 'rb') as f:
            return pickle.load(f)
