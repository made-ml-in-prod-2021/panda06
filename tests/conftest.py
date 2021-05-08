import os
from collections import OrderedDict
from typing import List

import pandas as pd
import pytest
from faker import Faker
from sklearn.ensemble import RandomForestClassifier

from ml_project.entities import FeatureParams
from ml_project.features import FeatureExtractor
from ml_project.models import Model


@pytest.fixture
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "data/heart.csv")


@pytest.fixture
def fake_data() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(42)
    dataset_size = 50
    data = {
        "age": [fake.pyint(min_value=29, max_value=77)
                for _ in range(dataset_size)],
        "sex": fake.random_elements(elements=OrderedDict([(0, 0.5),
                                                          (1, 0.5)]),
                                    length=dataset_size),
        "cp": [fake.pyint(min_value=0, max_value=3)
               for _ in range(dataset_size)],
        "trestbps": [fake.pyint(min_value=94, max_value=200)
                     for _ in range(dataset_size)],
        "chol": [fake.pyint(min_value=126, max_value=564)
                 for _ in range(dataset_size)],
        "fbs": fake.random_elements(elements=OrderedDict([(0, 0.5),
                                                          (1, 0.5)]),
                                    length=dataset_size),
        "restecg": [fake.pyint(min_value=0, max_value=2)
                    for _ in range(dataset_size)],
        "thalach": [fake.pyint(min_value=71, max_value=202)
                    for _ in range(dataset_size)],
        "exang": fake.random_elements(elements=OrderedDict([(0, 0.5),
                                                            (1, 0.5)]),
                                      length=dataset_size),
        "oldpeak": [fake.pyfloat(min_value=0, max_value=6)
                    for _ in range(dataset_size)],
        "slope": [fake.pyint(min_value=0, max_value=2)
                  for _ in range(dataset_size)],
        "ca": [fake.pyint(min_value=0, max_value=4)
               for _ in range(dataset_size)],
        "thal": [fake.pyint(min_value=0, max_value=3)
                 for _ in range(dataset_size)],
        "target": fake.random_elements(elements=OrderedDict([(0, 0.5),
                                                             (1, 0.5)]),
                                       length=dataset_size)
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def serialized_model_path() -> str:
    return "tests/model.pkl"


@pytest.fixture
def serialized_extractor_path() -> str:
    return "tests/extractor.pkl"


@pytest.fixture
def target_col() -> str:
    return "target"


@pytest.fixture
def categorical_features() -> List[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@pytest.fixture
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture
def features_to_drop() -> List[str]:
    return []


@pytest.fixture
def feature_params(categorical_features: List[str],
                   numerical_features: List[str],
                   features_to_drop: List[str], target_col: str):
    return FeatureParams(categorical_features,
                         numerical_features,
                         features_to_drop,
                         target_col)


@pytest.fixture
def extractor(feature_params) -> FeatureExtractor:
    extractor = FeatureExtractor(feature_params)
    return extractor


@pytest.fixture
def model(feature_params) -> Model:
    model = Model(RandomForestClassifier())
    return model
