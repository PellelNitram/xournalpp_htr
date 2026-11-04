"""Geometric stroke features shared by the detectors.

Xournal++ strokes carry no per-stroke timestamps, so everything here is spatial
plus the stroke order in the document (the writing order).
"""

import numpy as np

from xournalpp_htr.training.word_detector_online.strokes import StrokePage

PAIR_FEATURE_NAMES = [
    "gap_x",
    "start_dx",
    "jump_x",
    "jump_y",
    "center_dy",
    "overlap_x",
    "overlap_y",
    "width_a",
    "height_a",
    "width_b",
    "height_b",
]


def bounding_boxes(page: StrokePage) -> np.ndarray:
    """Return an ``(n, 4)`` array of ``xmin, ymin, xmax, ymax`` per stroke."""
    return np.array([[x.min(), y.min(), x.max(), y.max()] for x, y in page.strokes])


def page_scale(boxes: np.ndarray) -> float:
    """Typical stroke size of a page, used to make distances scale-free."""
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    return float(max(np.median(sizes), 1e-6))


def _overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> float:
    """Overlap of two intervals relative to the shorter one, in ``[0, 1]``."""
    shorter = max(min(hi_a - lo_a, hi_b - lo_b), 1e-6)
    return float(np.clip((min(hi_a, hi_b) - max(lo_a, lo_b)) / shorter, 0.0, 1.0))


def consecutive_pair_features(page: StrokePage) -> np.ndarray:
    """Features of each pair of strokes ``(i, i + 1)``, see `PAIR_FEATURE_NAMES`.

    Shape ``(n - 1, len(PAIR_FEATURE_NAMES))``. All lengths are divided by the
    page's typical stroke size.
    """
    boxes = bounding_boxes(page)
    scale = page_scale(boxes)
    rows = []
    for i in range(len(page.strokes) - 1):
        a, b = boxes[i], boxes[i + 1]
        ax, ay = page.strokes[i]
        bx, by = page.strokes[i + 1]
        rows.append(
            [
                (b[0] - a[2]) / scale,
                (b[0] - a[0]) / scale,
                (bx[0] - ax[-1]) / scale,
                (by[0] - ay[-1]) / scale,
                ((b[1] + b[3]) - (a[1] + a[3])) / 2 / scale,
                _overlap(a[0], a[2], b[0], b[2]),
                _overlap(a[1], a[3], b[1], b[3]),
                (a[2] - a[0]) / scale,
                (a[3] - a[1]) / scale,
                (b[2] - b[0]) / scale,
                (b[3] - b[1]) / scale,
            ]
        )
    return np.array(rows).reshape(-1, len(PAIR_FEATURE_NAMES))
