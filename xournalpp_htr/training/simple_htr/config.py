"""Hydra structured config for SimpleHTR training and inference.

All model/training constants live here as a single source of truth.
Training uses ``@hydra.main`` to parse overrides from the CLI or YAML;
other scripts (infer, export, demo) import the defaults directly.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    input_height: int = 32
    input_width: int = 128
    cnn_channels: tuple = (32, 64, 128, 128, 256)
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    epoch_max: int = 100
    patience_max: int = 25
    val_epoch: int = 1
    num_workers: int = 4


@dataclass
class DataConfig:
    data_path: str | None = None
    cache_path: str = "dataset_cache.pickle"
    percent_train_data: int = 95
    shuffle_data_loader: bool = True


@dataclass
class SeedConfig:
    split: int = 42
    model: int = 1337


@dataclass
class AugmentationConfig:
    enabled: bool = False


@dataclass
class SimpleHTRConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    output_path: str = "outputs"
