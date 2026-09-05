"""Hydra structured config for YOLO word detector training and inference.

All model/training constants live here as a single source of truth.
Training uses ``@hydra.main`` to parse overrides from the CLI or YAML;
other scripts (export, demo) import the defaults directly.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    variant: str = "yolov8s.pt"
    imgsz: int = 1024


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch: int = 16
    patience: int = 10
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    warmup_epochs: int = 3
    workers: int = 8
    save_period: int = 5
    device: str = "0"


@dataclass
class AugmentationConfig:
    mosaic: float = 0.5
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.2
    degrees: float = 2.0
    translate: float = 0.1
    scale: float = 0.3
    fliplr: float = 0.0
    flipud: float = 0.0


@dataclass
class InferenceConfig:
    conf: float = 0.25
    imgsz: int = 1024


@dataclass
class DataConfig:
    dataset_dir: str = "dataset"
    val_split: float = 0.15


@dataclass
class SeedConfig:
    split: int = 42


@dataclass
class WordDetectorYOLOConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    output_path: str = "outputs"
