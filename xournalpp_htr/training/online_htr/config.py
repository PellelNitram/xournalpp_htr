"""Hydra structured config for OnlineHTR training and export.

All model/training constants live here as a single source of truth.
Training uses ``@hydra.main`` to parse overrides from the CLI or YAML;
other scripts (infer, export) import the defaults directly.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    # Reference (Carbune et al. 2020 / OnlineHTR): 3 bidirectional LSTM layers.
    nodes_per_layer: int = 64
    number_of_layers: int = 3
    dropout: float = 0.25
    # Input channels (dx, dy, n); there is no time channel, see features.py.
    input_channels: int = 3


@dataclass
class FeatureConfig:
    # Strokes are resampled to this many points per unit of ink height.
    points_per_unit: float = 20.0


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    epoch_max: int = 300
    patience_max: int = 25
    val_epoch: int = 1
    num_workers: int = 4
    grad_clip: float = 5.0
    # ReduceLROnPlateau on the validation CER.
    lr_factor: float = 0.5
    lr_patience: int = 5


@dataclass
class DataConfig:
    # IAM-OnDB root (lineStrokes-all/, ascii-all/, forms.txt). `None` resolves
    # it from HF Hub (needs HF_TOKEN).
    data_path: str | None = None
    # "line": whole text lines; "word": words cut from lines at stroke gaps
    # (the pipeline recognises words); "both": the union.
    level: str = "both"
    val_writer_fraction: float = 0.1
    split_seed: int = 42
    min_gap_ratio: float = 1.5
    # Load only this many lines if >= 0 (debugging).
    limit: int = -1
    shuffle_data_loader: bool = True


@dataclass
class AugmentationConfig:
    enabled: bool = True
    # Random x-scale in [1 - x_scale, 1 + x_scale].
    x_scale: float = 0.15
    # Random shear (x += s * y), s in [-shear, shear].
    shear: float = 0.3
    # Random rotation in degrees, in [-rotation, rotation].
    rotation: float = 3.0


@dataclass
class SeedConfig:
    model: int = 1337


@dataclass
class OnlineHTRConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    output_path: str = "outputs"


def _load_section(checkpoint_dir: Path, name: str) -> dict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    return OmegaConf.to_container(cfg[name])


def load_model_config(checkpoint_dir: Path) -> ModelConfig:
    """Load ``ModelConfig`` from a training run's ``config.yaml``."""
    return ModelConfig(**_load_section(checkpoint_dir, "model"))


def load_feature_config(checkpoint_dir: Path) -> FeatureConfig:
    """Load ``FeatureConfig`` from a training run's ``config.yaml``."""
    return FeatureConfig(**_load_section(checkpoint_dir, "features"))
