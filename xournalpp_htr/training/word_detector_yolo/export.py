"""Export a trained YOLO word detector checkpoint to ONNX.

Produces ``model.onnx`` and ``config.json`` for use with
:class:`xournalpp_htr.inference_models.YOLOWordDetectorModel`.

Usage::

    uv run python -m xournalpp_htr.training.word_detector_yolo.export \\
        --checkpoint runs/detect/train_xxx/weights/best.pt --output-dir exports/
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from xournalpp_htr.training.word_detector_yolo.config import InferenceConfig

_INFERENCE_DEFAULTS = InferenceConfig()

HF_REPO_ID = "PellelNitram/xournalpp-htr-word-detector-yolo"


def build_config(checkpoint: Path) -> dict:
    return {
        "checkpoint": str(checkpoint),
        "model_name": "word_detector_yolo",
        "conf": _INFERENCE_DEFAULTS.conf,
        "imgsz": _INFERENCE_DEFAULTS.imgsz,
        "names": {0: "word"},
    }


def export(checkpoint: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(checkpoint))
    exported_path = model.export(format="onnx", imgsz=_INFERENCE_DEFAULTS.imgsz)

    onnx_dst = output_dir / "model.onnx"
    shutil.move(str(exported_path), str(onnx_dst))

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(build_config(checkpoint), f, indent=2)

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
        help="Path to the trained YOLO .pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory to write model.onnx and config.json into.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After export, upload to HF Hub (requires authentication).",
    )
    args = parser.parse_args()

    export(args.checkpoint, args.output_dir)
    if args.upload:
        upload_to_hub(args.output_dir)


if __name__ == "__main__":
    main()
