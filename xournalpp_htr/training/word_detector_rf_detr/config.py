"""Hydra structured config for RF-DETR word detector training and inference.

All model/training constants live here as a single source of truth.
Training uses ``@hydra.main`` to parse overrides from the CLI or YAML;
other scripts (export, demo) import the defaults directly.
"""

from dataclasses import dataclass, field

#: RF-DETR positional embeddings require the input resolution to be a
#: multiple of this value.
RESOLUTION_DIVISOR = 56


@dataclass
class ModelConfig:
    #: Model size. Resolved at runtime to the ``rfdetr.RFDETR<Variant>``
    #: class, so which values are valid depends on the installed rfdetr
    #: version -- newer ones dropped ``base`` in favour of the
    #: nano/small/medium/large lineup. ``medium`` is the closest successor to
    #: the old ``base`` default. Check what your install offers with
    #: ``model_factory.available_variants()``; the ``seg*`` entries are
    #: segmentation models and ``keypointpreview`` does keypoints, so neither
    #: is a drop-in word-box detector.
    variant: str = "medium"
    #: Must be divisible by ``RESOLUTION_DIVISOR``. 1008 = 56 * 18, chosen as
    #: the closest analogue to the YOLO detector's imgsz=1024.
    resolution: int = 1008


@dataclass
class TrainingConfig:
    epochs: int = 50
    #: Effective batch size is ``batch_size * grad_accum_steps``; RF-DETR is
    #: tuned for a total of 16.
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    weight_decay: float = 1e-4
    early_stopping: bool = True
    early_stopping_patience: int = 10
    checkpoint_interval: int = 5
    workers: int = 8
    device: str = "cuda"


@dataclass
class InferenceConfig:
    threshold: float = 0.5
    resolution: int = 1008


@dataclass
class DataConfig:
    dataset_dir: str = "dataset"
    val_split: float = 0.15


@dataclass
class SeedConfig:
    split: int = 42


@dataclass
class WordDetectorRFDETRConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    output_path: str = "outputs"
