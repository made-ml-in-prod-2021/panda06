import os
from typing import List
from dacite import from_dict
import pytest
from omegaconf import OmegaConf, DictConfig

from ml_project.eval import eval_model
from ml_project.entities import EvalParams


@pytest.fixture
def eval_params(dataset_path: str) -> EvalParams:

    params_dict = {
        "data_path": dataset_path,
        "model_path": "model.pkl",
        "extractor_path": "extractor.pkl",
        "output_data_path": "predicts.csv"

    }
    return from_dict(data_class=EvalParams, data=params_dict)


def test_eval_full_pipeline(eval_params: EvalParams):
    eval_model(eval_params)
    assert os.path.exists(eval_params.output_data_path)

