from dataclasses import dataclass, field


@dataclass()
class EvalParams:
    model_path: str = field(default=None)
    data_path: str = field(default=None)
    output_data_path: str = field(default=None)
    extractor_path: str = field(default=None)
