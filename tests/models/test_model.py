import os
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from ml_project.features import FeatureExtractor
from ml_project.models import Model


@pytest.fixture
def preprocess_data(fake_data: pd.DataFrame,
                    extractor: FeatureExtractor) -> Tuple[pd.DataFrame,
                                                          pd.Series]:
    transformer = extractor.build_transformer()
    X = transformer.fit_transform(fake_data)
    target = extractor.extract_target(fake_data)
    return X, target


def test_serialize_and_deserialize_model(serialized_model_path: str):
    model = Model(RandomForestClassifier())
    model.serialize(serialized_model_path)
    assert os.path.exists(serialized_model_path)
    loaded_model = Model.deserialize(serialized_model_path)
    assert isinstance(loaded_model, Model)
    assert hasattr(loaded_model, "model") and \
           isinstance(loaded_model.model, RandomForestClassifier)


def test_evaluate():
    targets = np.random.randint(2, size=10)
    predicts = np.random.uniform(low=0, high=1, size=10)
    metrics = Model.evaluate_model(predicts, targets)
    assert isinstance(metrics, dict)
    assert all(0 <= x <= 1 for x in metrics.values()), metrics


def test_train_model(model: Model,
                     preprocess_data: Tuple[pd.DataFrame, pd.Series]):
    X, y = preprocess_data
    model.train_model(X, y)
    try:
        model.predict(X)
        assert True
    except NotFittedError:
        assert False


def test_predict_model(model: Model,
                       preprocess_data: Tuple[pd.DataFrame, pd.Series]):
    X, y = preprocess_data
    model.train_model(X, y)
    predict = model.predict(X)
    assert y.shape == predict.shape
