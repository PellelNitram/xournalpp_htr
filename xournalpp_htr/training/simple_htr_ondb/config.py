"""Hydra structured config for SimpleHTR-OnDB training and inference.

Copy of ``simple_htr/config.py`` with the defaults of its best run (experiment 3:
dropout 0.5, augmentation on, lr 0.0005) and the data settings for rendered
IAM-OnDB crops.

All model/training constants live here as a single source of truth.
Training uses ``@hydra.main`` to parse overrides from the CLI or YAML;
other scripts (infer, export, demo) import the defaults directly.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    input_height: int = 32
    input_width: int = 128
    cnn_channels: tuple = (32, 64, 128, 128, 256)
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    learning_rate: float = 0.0005
    batch_size: int = 64
    epoch_max: int = 200
    patience_max: int = 25
    val_epoch: int = 1
    num_workers: int = 4


@dataclass
class DataConfig:
    # Output of `prepare_data` (images/ and manifest.jsonl). `None` means the
    # gitignored `data/` folder next to this file. The train/val split is fixed
    # by writer in the manifest.
    data_path: str | None = None
    shuffle_data_loader: bool = True
    # Training sources. Both on trains on the union; their val splits are
    # concatenated too, and checkpoints are selected on the combined val CER.
    use_ondb: bool = True
    use_iam: bool = True
    # IAM-DB root (words/, ascii/). `None` resolves it from HF Hub.
    iam_data_path: str | None = None
    iam_val_fraction: float = 0.05
    iam_split_seed: int = 42


@dataclass
class SeedConfig:
    model: int = 1337


@dataclass
class AugmentationConfig:
    enabled: bool = True


@dataclass
class SimpleHTROnDBConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    output_path: str = "outputs"


def load_model_config(checkpoint_dir: Path) -> ModelConfig:
    """Load ``ModelConfig`` from a training run's ``config.yaml``."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    return ModelConfig(**OmegaConf.to_container(cfg.model))
