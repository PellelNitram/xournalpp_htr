"""Hydra structured config for WordDetector training and inference.

All model/detection constants live here as a single source of truth.
Training uses ``@hydra.main`` to parse overrides from the CLI or YAML;
other scripts (infer, export, demo) import the defaults directly.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    input_height: int = 448
    input_width: int = 448
    output_height: int = 224
    output_width: int = 224


@dataclass
class DetectionConfig:
    fg_threshold: float = 0.5
    max_detections: int = 1000


@dataclass
class NormalizationConfig:
    scale: float = 255.0
    shift: float = -0.5


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epoch_max: int = 3
    patience_max: int = 50
    val_epoch: int = 1
    num_workers: int = 1


@dataclass
class DataConfig:
    data_path: str | None = None
    cache_path: str = "dataset_cache.pickle"
    percent_train_data: int = 80
    shuffle_data_loader: bool = True


@dataclass
class SeedConfig:
    split: int = 42
    model: int = 1337


@dataclass
class WordDetectorConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    output_path: str = "outputs"
