"""Evaluate a WordDetector checkpoint on the xournalpp_htr benchmark dataset.

Self-contained detection-only benchmark. Loads the benchmark dataset from
HuggingFace Hub, runs the detector on each page, and reports detection
precision/recall using the same matching logic as ``xournalpp_htr.benchmark``.

Usage::

    uv run python -m xournalpp_htr.training.word_detector.eval_benchmark \
        --checkpoint experiments/experiment3/augtrue_seed44/best_model.pth
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from xournalpp_htr.documents import get_document
from xournalpp_htr.training.word_detector.infer import run_image_through_network
from xournalpp_htr.xio import load_benchmark

RENDER_DPI = 150

_MIN_COVERAGE = 0.8
_MAX_AREA_RATIO = 8.0


def _parse_gt_words(gt_path, document):
    with open(gt_path) as f:
        gt = json.load(f)

    text_classes = {"word", "digit", "mathematical_expression"}
    words = []
    for ann in gt["annotations"]:
        if ann["class"] not in text_classes:
            continue
        page_index = ann["page_index"]
        layer_index = ann["layer_index"]
        layer = document.pages[page_index].layers[layer_index]
        xs, ys = [], []
        for idx in ann["stroke_indices"]:
            stroke = layer.strokes[idx]
            xs.extend(stroke.x.tolist())
            ys.extend(stroke.y.tolist())
        words.append(
            {
                "xmin": float(np.min(xs)),
                "xmax": float(np.max(xs)),
                "ymin": float(np.min(ys)),
                "ymax": float(np.max(ys)),
                "page_index": page_index,
            }
        )
    return words


def _matches(gt, pred):
    ix1 = max(gt["xmin"], pred["xmin"])
    iy1 = max(gt["ymin"], pred["ymin"])
    ix2 = min(gt["xmax"], pred["xmax"])
    iy2 = min(gt["ymax"], pred["ymax"])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_gt = (gt["xmax"] - gt["xmin"]) * (gt["ymax"] - gt["ymin"])
    area_pred = (pred["xmax"] - pred["xmin"]) * (pred["ymax"] - pred["ymin"])
    if area_gt == 0.0:
        return False
    return (
        intersection / area_gt >= _MIN_COVERAGE
        and area_pred / area_gt <= _MAX_AREA_RATIO
    )


def _iou(gt, pred):
    ix1 = max(gt["xmin"], pred["xmin"])
    iy1 = max(gt["ymin"], pred["ymin"])
    ix2 = min(gt["xmax"], pred["xmax"])
    iy2 = min(gt["ymax"], pred["ymax"])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0.0:
        return 0.0
    area_gt = (gt["xmax"] - gt["xmin"]) * (gt["ymax"] - gt["ymin"])
    area_pred = (pred["xmax"] - pred["xmin"]) * (pred["ymax"] - pred["ymin"])
    return intersection / (area_gt + area_pred - intersection)


def _match(gt_words, predictions):
    candidates = []
    for gt in gt_words:
        for pred in predictions.get(gt["page_index"], []):
            if _matches(gt, pred):
                candidates.append((_iou(gt, pred), gt, pred))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt, matched_pred = set(), set()
    pairs = []
    for _, gt, pred in candidates:
        gt_id, pred_id = id(gt), id(pred)
        if gt_id not in matched_gt and pred_id not in matched_pred:
            pairs.append((gt, pred))
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
    return pairs


def detect_page(img_gray, checkpoint, device):
    result = run_image_through_network(
        image_grayscale=img_gray,
        model_path=checkpoint,
        device=device,
    )
    scaling_factors = np.array(img_gray.shape) / np.array(
        result["model_input_image"].shape
    )
    return [aabb.scale(*scaling_factors[::-1]) for aabb in result["aabbs"]]


def run(checkpoint, device):
    samples = load_benchmark()

    total_gt = 0
    total_pred = 0
    total_matched = 0

    for sample in samples:
        document = get_document(sample.xopp_path)
        gt_words = _parse_gt_words(sample.gt_path, document)

        predictions = {}
        coord_scale = document.DPI / RENDER_DPI

        for page_index in range(len(document.pages)):
            if (
                len(document.pages[page_index].layers) == 0
                or len(document.pages[page_index].layers[0].strokes) == 0
            ):
                predictions[page_index] = []
                continue

            with tempfile.NamedTemporaryFile(
                dir="/tmp",
                delete=False,
                prefix=f"eval_benchmark__page{page_index}__",
                suffix=".jpg",
            ) as tmpfile:
                tmp_path = Path(tmpfile.name)

            document.save_page_as_image(page_index, tmp_path, False, dpi=RENDER_DPI)
            img = cv2.imread(str(tmp_path), cv2.IMREAD_GRAYSCALE)
            tmp_path.unlink(missing_ok=True)

            boxes = detect_page(img, checkpoint, device)
            predictions[page_index] = [
                {
                    "xmin": b.x_min * coord_scale,
                    "xmax": b.x_max * coord_scale,
                    "ymin": b.y_min * coord_scale,
                    "ymax": b.y_max * coord_scale,
                }
                for b in boxes
            ]

        n_pred = sum(len(v) for v in predictions.values())
        pairs = _match(gt_words, predictions)

        total_gt += len(gt_words)
        total_pred += n_pred
        total_matched += len(pairs)

        sample_precision = len(pairs) / n_pred if n_pred > 0 else 0.0
        sample_recall = len(pairs) / len(gt_words) if gt_words else 0.0
        print(
            f"  {sample.xopp_path.name}: "
            f"precision={sample_precision:.3f} recall={sample_recall:.3f} "
            f"({len(pairs)}/{len(gt_words)} matched, {n_pred} predicted)"
        )

    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print()
    print(f"Detection benchmark results ({checkpoint}):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Matched:   {total_matched}/{total_gt} GT words")
    print(f"  Predicted: {total_pred} boxes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained .pth checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help='Inference device. "auto" selects GPU if available.',
    )
    args = parser.parse_args()
    run(args.checkpoint, args.device)


if __name__ == "__main__":
    main()
