import os

import pytest
from dacite import from_dict

from ml_project.entities import EvalParams
from ml_project.eval import eval_model


@pytest.fixture
def eval_params(test_dataset_path: str) -> EvalParams:

    params_dict = {
        "data_path": test_dataset_path,
        "model_path": "model.pkl",
        "extractor_path": "extractor.pkl",
        "output_data_path": "predicts.csv"

    }
    return from_dict(data_class=EvalParams, data=params_dict)


@pytest.mark.order(2)
def test_eval_full_pipeline(eval_params: EvalParams):
    eval_model(eval_params)
    assert os.path.exists(eval_params.output_data_path)
