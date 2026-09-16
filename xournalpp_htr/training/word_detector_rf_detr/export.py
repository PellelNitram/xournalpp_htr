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
    return {
        "checkpoint": str(checkpoint),
        "model_name": "word_detector_rf_detr",
        "variant": variant,
        "threshold": _INFERENCE_DEFAULTS.threshold,
        "resolution": resolution,
        "names": {0: "word"},
    }


def export(
    checkpoint: Path,
    output_dir: Path,
    variant: str = _MODEL_DEFAULTS.variant,
    resolution: int = _MODEL_DEFAULTS.resolution,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(variant, resolution, pretrain_weights=str(checkpoint))

    # RF-DETR names the artefact itself, so export into a scratch directory
    # and pick up whatever .onnx it produced.
    with tempfile.TemporaryDirectory() as tmp:
        model.export(output_dir=tmp)
        produced = sorted(Path(tmp).glob("*.onnx"))
        if not produced:
            raise RuntimeError(f"RF-DETR export produced no .onnx file in {tmp}")
        if len(produced) > 1:
            print(f"Note: multiple ONNX files produced, using {produced[0].name}")
        onnx_dst = output_dir / "model.onnx"
        shutil.move(str(produced[0]), str(onnx_dst))

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
