"""Run word detection on an image using a trained YOLO model.

Usage:
    uv run python predict.py photo.jpg
    uv run python predict.py photo.jpg --weights runs/detect/train/weights/best.pt
    uv run python predict.py photo.jpg --conf 0.3 --save-txt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def draw_predictions(img: np.ndarray, results) -> np.ndarray:
    annotated = img.copy()
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect handwritten words in an image")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/train/weights/best.pt"),
        help="Path to trained model weights",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--save-txt", action="store_true", help="Save detections as YOLO-format .txt"
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Model weights not found at {args.weights}")

    model = YOLO(str(args.weights))
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    results = model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device)
    n_detections = len(results[0].boxes)
    print(f"Detected {n_detections} word(s).")

    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)

    annotated = draw_predictions(img, results)
    out_path = out_dir / f"{args.image.stem}_pred.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to {out_path}")

    if args.save_txt:
        txt_path = out_dir / f"{args.image.stem}_pred.txt"
        h, w = img.shape[:2]
        lines = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            conf = float(box.conf[0])
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")
        txt_path.write_text("\n".join(lines) + "\n")
        print(f"Saved detections to {txt_path}")


if __name__ == "__main__":
    main()
