import pandas as pd

from ml_project.data.make_dataset import read_data, split_train_val_data
from ml_project.entities import SplittingParams


def test_load_dataset(dataset_path: str):
    data = read_data(dataset_path)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_split_dataset(dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=239, val_size=val_size,)
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > val.shape[0]
    assert train.shape[1] == val.shape[1]
    assert train.shape[0] + val.shape[0] == data.shape[0]
    assert isinstance(train, pd.DataFrame) and isinstance(val, pd.DataFrame)
    assert not (train.empty or val.empty)
