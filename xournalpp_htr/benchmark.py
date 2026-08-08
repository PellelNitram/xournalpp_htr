import json
from dataclasses import dataclass

import numpy as np

from xournalpp_htr.documents import get_document
from xournalpp_htr.models import PageIndex, WordPrediction, compute_predictions
from xournalpp_htr.xio import load_benchmark

# Annotation classes that carry a text transcription (per ground_truth.schema.json).
_TEXT_CLASSES = {"word", "digit", "mathematical_expression"}

# A prediction matches a ground truth word when it covers most of that word and
# is not wildly larger than it.
#
# IoU is deliberately not used as the criterion. Word detectors pad their boxes
# (see the `margin` in `compute_predictions`), and because IoU divides by the
# union, that padding costs little on long words but is fatal on short ones: a
# perfectly centred box around "is" scores ~0.2 purely because the word is
# small. Coverage asks the question we actually care about -- did the pipeline
# find this word? -- and is unaffected by padding. The area cap stops a single
# oversized box, such as one spanning a whole line, from counting as a match.
_MIN_COVERAGE = 0.8  # fraction of the GT box that must lie inside the prediction
_MAX_AREA_RATIO = 8.0  # prediction area may exceed the GT area by at most this


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
    cer_case_insensitive: float
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
    """Intersection over union. Ranks candidate matches, it does not accept them."""
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


def _matches(a: GroundTruthWord, b: WordPrediction) -> bool:
    """Whether a prediction covers enough of a GT word to count as finding it."""
    ix1 = max(a.xmin, b.xmin)
    iy1 = max(a.ymin, b.ymin)
    ix2 = min(a.xmax, b.xmax)
    iy2 = min(a.ymax, b.ymax)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    if area_a == 0.0:
        return False
    return intersection / area_a >= _MIN_COVERAGE and area_b / area_a <= _MAX_AREA_RATIO


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
    """Greedily pair ground truth words with predictions, one to one per page.

    Acceptable pairs are decided by `_matches`; among those, the pair with the
    tightest fit (highest IoU) is taken first so that a prediction goes to the
    word it sits on rather than to a neighbour it merely overlaps.
    """
    candidates = []
    for gt in gt_words:
        for pred in predictions.get(gt.page_index, []):
            if _matches(gt, pred):
                candidates.append((_iou(gt, pred), gt, pred))

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
    total_edit_chars_case_insensitive = 0
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
            total_edit_chars_case_insensitive += round(
                _cer(gt_word.text.lower(), pred_word.text.lower()) * len(gt_word.text)
            )

    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    cer = (
        total_edit_chars / total_gt_chars_matched if total_gt_chars_matched > 0 else 0.0
    )
    cer_case_insensitive = (
        total_edit_chars_case_insensitive / total_gt_chars_matched
        if total_gt_chars_matched > 0
        else 0.0
    )

    return BenchmarkResult(
        precision=precision,
        recall=recall,
        cer=cer,
        cer_case_insensitive=cer_case_insensitive,
        n_gt_words=total_gt,
        n_predicted_words=total_pred,
        n_matched=total_matched,
    )
