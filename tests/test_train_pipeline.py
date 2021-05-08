import os
from typing import List

import pytest
from omegaconf import OmegaConf, DictConfig

from ml_project.train import train_model


@pytest.fixture
def train_params(categorical_features: List[str],
                 numerical_features: List[str],
                 features_to_drop: List[str],
                 target_col: str, dataset_path: str) -> DictConfig:

    params_dict = {
        "model": {
            "_target_": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 100
        },
        "splitting_params": {
            "val_size": 0.2,
            "random_state": 42
        },
        "feature_params": {
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "features_to_drop": features_to_drop,
            "target_column": target_col
        },
        "general": {
            "data_path": dataset_path,
            "model_output_path": "model.pkl",
            "extractor_output_path": "extractor.pkl",
            "metrics_path": "metrics.json"
        }

    }
    return OmegaConf.create(params_dict)


@pytest.mark.order(1)
def test_train_full_pipeline(train_params: DictConfig):
    train_model(train_params)
    assert os.path.exists(train_params.general.model_output_path)
    assert os.path.exists(train_params.general.extractor_output_path)
    assert os.path.exists(train_params.general.metrics_path)
