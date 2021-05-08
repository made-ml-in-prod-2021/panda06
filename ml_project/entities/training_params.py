from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    load_path: str = field(default=None)
    output_path: str = field(default=None)
