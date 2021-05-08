from dataclasses import dataclass, field
from .feature_params import FeatureParams


@dataclass()
class EvalParams:
    model_path: str = field(default=None)
    data_path: str = field(default=None)
    output_data_path: str = field(default=None)
    extractor_path: str = field(default=None)
