from dataclasses import dataclass

from .data_params import DataParams
from .feature_params import FeatureParams
from .split_params import SplittingParams
from .training_params import TrainingParams


@dataclass()
class Params:
    data_params: DataParams = DataParams()
    feature_params: FeatureParams = FeatureParams()
    splitting_params: SplittingParams = SplittingParams()
    training_params: TrainingParams = TrainingParams()
