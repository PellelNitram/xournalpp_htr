"""Run word detection on an image using a trained RF-DETR model.

Usage:
    uv run python predict.py photo.jpg
    uv run python predict.py photo.jpg --weights outputs/train_xxx/checkpoint_best_total.pth
    uv run python predict.py photo.jpg --threshold 0.3 --save-txt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from xournalpp_htr.training.word_detector_rf_detr.config import (
    InferenceConfig,
    ModelConfig,
)
from xournalpp_htr.training.word_detector_rf_detr.model_factory import build_model

_INFERENCE_DEFAULTS = InferenceConfig()
_MODEL_DEFAULTS = ModelConfig()


def draw_predictions(img: np.ndarray, detections) -> np.ndarray:
    """Annotate ``img`` with the boxes of a supervision ``Detections``."""
    annotated = img.copy()
    for (x1, y1, x2, y2), conf in zip(
        detections.xyxy.astype(int), detections.confidence, strict=False
    ):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"{float(conf):.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return annotated


def load_model(weights: Path, variant: str, resolution: int):
    return build_model(variant, resolution, pretrain_weights=str(weights))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect handwritten words in an image")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/train/checkpoint_best_total.pth"),
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_INFERENCE_DEFAULTS.threshold,
        help="Confidence threshold",
    )
    parser.add_argument("--variant", type=str, default=_MODEL_DEFAULTS.variant)
    parser.add_argument(
        "--resolution", type=int, default=_INFERENCE_DEFAULTS.resolution
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="Save detections as YOLO-format .txt"
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Model weights not found at {args.weights}")

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model = load_model(args.weights, args.variant, args.resolution)
    # RF-DETR expects RGB; cv2.imread gives BGR.
    detections = model.predict(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=args.threshold
    )
    print(f"Detected {len(detections)} word(s).")

    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)

    annotated = draw_predictions(img, detections)
    out_path = out_dir / f"{args.image.stem}_pred.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to {out_path}")

    if args.save_txt:
        txt_path = out_dir / f"{args.image.stem}_pred.txt"
        h, w = img.shape[:2]
        lines = []
        for (x1, y1, x2, y2), conf in zip(
            detections.xyxy, detections.confidence, strict=False
        ):
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(conf):.4f}")
        txt_path.write_text("\n".join(lines) + "\n")
        print(f"Saved detections to {txt_path}")


if __name__ == "__main__":
    main()
