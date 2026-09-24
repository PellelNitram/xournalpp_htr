"""Export a trained RF-DETR word detector checkpoint to ONNX.

Produces ``model.onnx`` and ``config.json`` for use with
:class:`xournalpp_htr.inference_models.RFDETRWordDetectorModel`.

Usage::

    uv run python -m xournalpp_htr.training.word_detector_rf_detr.export \\
        --checkpoint outputs/train_xxx/checkpoint_best_total.pth \\
        --output-dir exports/
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from xournalpp_htr.training.word_detector_rf_detr.config import (
    InferenceConfig,
    ModelConfig,
)
from xournalpp_htr.training.word_detector_rf_detr.model_factory import build_model

_INFERENCE_DEFAULTS = InferenceConfig()
_MODEL_DEFAULTS = ModelConfig()

HF_REPO_ID = "PellelNitram/xournalpp-htr-word-detector-rf-detr"


def build_config(checkpoint: Path, variant: str, resolution: int) -> dict:
    """Inference parameters written alongside the ONNX export.

    The exported classifier has two classes because the COCO conversion in
    ``train.py`` follows the Roboflow convention of a placeholder category 0
    alongside the real ``word`` category 1. RF-DETR does not carry those COCO
    ids through to the model: ``word`` lands on classifier index **0**, and
    index 1 is never trained (verified on the export — index 1 peaks at a 0.008
    score). ``num_select`` mirrors RF-DETR's own post-processor.
    """
    return {
        "checkpoint": str(checkpoint),
        "model_name": "word_detector_rf_detr",
        "variant": variant,
        "threshold": _INFERENCE_DEFAULTS.threshold,
        "resolution": resolution,
        "word_class_index": 0,
        "num_select": 300,
        "names": {0: "word", 1: "untrained (unused)"},
    }


def export(
    checkpoint: Path,
    output_dir: Path,
    variant: str = _MODEL_DEFAULTS.variant,
    resolution: int = _MODEL_DEFAULTS.resolution,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(variant, resolution, pretrain_weights=str(checkpoint))

    # RF-DETR names the artefact itself, so export into a scratch directory and
    # move what it returns into place. fp16 is disabled because inference runs
    # through onnxruntime on CPU (ADR 006), where fp16 support is poor.
    with tempfile.TemporaryDirectory() as tmp:
        produced = Path(model.export(output_dir=tmp, fp16=False))
        if not produced.exists():
            raise RuntimeError(
                f"RF-DETR export reported {produced}, which does not exist"
            )
        onnx_dst = output_dir / "model.onnx"
        shutil.move(str(produced), str(onnx_dst))

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(build_config(checkpoint, variant, resolution), f, indent=2)

    print(f"Wrote {onnx_dst} and {config_path}")
    return {"onnx": onnx_dst, "config": config_path}


def upload_to_hub(output_dir: Path, repo_id: str = HF_REPO_ID) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    for filename in ("model.onnx", "config.json"):
        api.upload_file(
            path_or_fileobj=str(output_dir / filename),
            path_in_repo=filename,
            repo_id=repo_id,
        )
    print(f"Uploaded model.onnx + config.json to {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained RF-DETR .pth checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory to write model.onnx and config.json into.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=_MODEL_DEFAULTS.variant,
        help="RF-DETR variant the checkpoint was trained with.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=_MODEL_DEFAULTS.resolution,
        help="Input resolution the checkpoint was trained with.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After export, upload to HF Hub (requires authentication).",
    )
    args = parser.parse_args()

    export(args.checkpoint, args.output_dir, args.variant, args.resolution)
    if args.upload:
        upload_to_hub(args.output_dir)


if __name__ == "__main__":
    main()
