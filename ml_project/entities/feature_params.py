from dataclasses import dataclass, field
from typing import List


@dataclass()
class FeatureParams:
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    features_to_drop: List[str] = field(default_factory=list)
    target_column: str = field(default="target")
