import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_project.features import FeatureExtractor
from ml_project.entities import FeatureParams

EPS = 1e-8


def test_serialize_and_deserialize_extractor(serialized_extractor_path: str,
                                             extractor: FeatureExtractor):
    extractor.serialize(serialized_extractor_path)
    assert os.path.exists(serialized_extractor_path)
    loaded_model = FeatureExtractor.deserialize(serialized_extractor_path)
    assert isinstance(loaded_model, FeatureExtractor)


def test_extract_target(fake_data: pd.DataFrame, target_col: str, extractor: FeatureExtractor):
    target = extractor.extract_target(fake_data)
    assert np.all(target == fake_data[target_col])


def test_build_transformer(extractor: FeatureExtractor):
    transformer = extractor.build_transformer()
    assert isinstance(transformer, ColumnTransformer)


def create_categorical_data(n_rows: int, n_cols: int, max_val: int) -> pd.DataFrame:
    fake = Faker()
    data = []
    for _ in range(n_rows):
        data.append(fake.random_elements(elements=OrderedDict([(i, 0.25) for i in range(max_val + 1)]),
                                         length=n_cols))
    return pd.DataFrame(data)


def create_numerical_data(n_rows: int, n_cols: int) -> pd.DataFrame:
    fake = Faker()
    data = []
    for _ in range(n_rows):
        data.append([fake.pyint(min_value=29, max_value=77) for _ in range(n_cols)])
    return pd.DataFrame(data)


def test_categorical_pipeline(extractor: FeatureExtractor):
    pipeline = extractor.build_categorical_pipeline()
    n_rows = 10
    n_cols = 2
    max_val = 4
    df = create_categorical_data(n_rows, n_cols, max_val)
    df = pipeline.fit_transform(df)
    assert df.shape == (n_rows, n_cols * max_val), df
    assert isinstance(pipeline, Pipeline)


def test_numerical__pipeline(extractor: FeatureExtractor):
    pipeline = extractor.build_numerical_pipeline()
    n_rows = 10
    n_cols = 4
    df = create_numerical_data(n_rows, n_cols)
    df = pipeline.fit_transform(df)
    assert isinstance(pipeline, Pipeline)
    assert df.shape == (n_rows, n_cols)
    means = np.mean(df, axis=0)
    stds = np.std(df, axis=0)
    assert np.all((-EPS <= means) & (means <= EPS))
    assert np.all((1 - EPS <= stds) & (stds <= 1 + EPS))


def test_fit_transform(extractor: FeatureExtractor, fake_data: pd.DataFrame, feature_params: FeatureParams):
    df = extractor.fit_transform(fake_data)
    expected_shape = (fake_data.shape[0],
                      len(feature_params.numerical_features) + \
                      fake_data[feature_params.categorical_features].nunique().sum()
    )
    assert df.shape == expected_shape
    means = np.mean(df, axis=0)
    assert np.all((-1 <= means) & (means <= 1))
