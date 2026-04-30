import json
from dataclasses import dataclass

import numpy as np

from xournalpp_htr.documents import get_document
from xournalpp_htr.models import PageIndex, WordPrediction, compute_predictions
from xournalpp_htr.xio import load_benchmark

# Annotation classes that carry a text transcription (per ground_truth.schema.json).
_TEXT_CLASSES = {"word", "digit", "mathematical_expression"}

# Minimum IoU to consider a prediction matched to a GT word.
_IOU_THRESHOLD = 0.5


@dataclass
class GroundTruthWord:
    text: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    page_index: int


@dataclass
class BenchmarkResult:
    precision: float
    recall: float
    cer: float
    n_gt_words: int
    n_predicted_words: int
    n_matched: int


def _load_gt_words(gt_path, document) -> list[GroundTruthWord]:
    with open(gt_path) as f:
        gt = json.load(f)

    words = []
    for ann in gt["annotations"]:
        if ann["class"] not in _TEXT_CLASSES:
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
            GroundTruthWord(
                text=ann["text"],
                xmin=float(np.min(xs)),
                xmax=float(np.max(xs)),
                ymin=float(np.min(ys)),
                ymax=float(np.max(ys)),
                page_index=page_index,
            )
        )
    return words


def _iou(a: GroundTruthWord, b: WordPrediction) -> float:
    ix1 = max(a.xmin, b.xmin)
    iy1 = max(a.ymin, b.ymin)
    ix2 = min(a.xmax, b.xmax)
    iy2 = min(a.ymax, b.ymax)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0.0:
        return 0.0
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    return intersection / (area_a + area_b - intersection)


def _cer(reference: str, hypothesis: str) -> float:
    """Character error rate between two strings via edit distance."""
    r, h = list(reference), list(hypothesis)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / max(len(r), 1)


def _match(
    gt_words: list[GroundTruthWord],
    predictions: dict[PageIndex, list[WordPrediction]],
) -> list[tuple[GroundTruthWord, WordPrediction]]:
    """Greedy IoU matching: highest IoU pairs are matched first."""
    candidates = []
    for gt in gt_words:
        for pred in predictions.get(gt.page_index, []):
            iou = _iou(gt, pred)
            if iou >= _IOU_THRESHOLD:
                candidates.append((iou, gt, pred))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt, matched_pred = set(), set()
    pairs = []
    for _, gt, pred in candidates:
        if id(gt) not in matched_gt and id(pred) not in matched_pred:
            pairs.append((gt, pred))
            matched_gt.add(id(gt))
            matched_pred.add(id(pred))
    return pairs


def run_benchmark(pipeline_name: str) -> BenchmarkResult:
    samples = load_benchmark()

    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_edit_chars = 0
    total_gt_chars_matched = 0

    for sample in samples:
        document = get_document(sample.xopp_path)
        gt_words = _load_gt_words(sample.gt_path, document)
        predictions = compute_predictions(pipeline_name, document)

        n_pred = sum(len(v) for v in predictions.values())
        pairs = _match(gt_words, predictions)

        total_gt += len(gt_words)
        total_pred += n_pred
        total_matched += len(pairs)

        for gt_word, pred_word in pairs:
            total_gt_chars_matched += len(gt_word.text)
            total_edit_chars += round(
                _cer(gt_word.text, pred_word.text) * len(gt_word.text)
            )

    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    cer = (
        total_edit_chars / total_gt_chars_matched if total_gt_chars_matched > 0 else 0.0
    )

    return BenchmarkResult(
        precision=precision,
        recall=recall,
        cer=cer,
        n_gt_words=total_gt,
        n_predicted_words=total_pred,
        n_matched=total_matched,
    )
