import pickle

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_project.entities import FeatureParams
from .normalizer import Normalizer


class FeatureExtractor(TransformerMixin):
    def __init__(self, params: FeatureParams):
        self.numerical_features = list(params.numerical_features)
        self.categorical_features = list(params.categorical_features)
        print(self.numerical_features, self.categorical_features)
        self.target_col = params.target_column
        self.transformer = self.build_transformer()

    def build_transformer(self) -> ColumnTransformer:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    self.build_categorical_pipeline(),
                    self.categorical_features,
                ),
                (
                    "numerical_pipeline",
                    self.build_numerical_pipeline(),
                    self.numerical_features,
                ),
            ]
        )
        return transformer

    @staticmethod
    def build_categorical_pipeline() -> Pipeline:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                ("ohe", OneHotEncoder()),
            ]
        )
        return categorical_pipeline

    @staticmethod
    def build_numerical_pipeline() -> Pipeline:
        num_pipeline = Pipeline(
            [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
             ("normalizer", Normalizer())]
        )
        return num_pipeline

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        self.transformer.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.transformer.transform(df))

    def extract_target(self, df: pd.DataFrame) -> pd.Series:
        target = df[self.target_col]
        return target

    def serialize(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize(path: str) -> "FeatureExtractor":
        with open(path, 'rb') as f:
            return pickle.load(f)
