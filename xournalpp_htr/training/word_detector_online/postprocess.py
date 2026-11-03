"""Clean-up of the word groups a detector predicts.

Merges small fragments of a word back into their neighbour and drops lone
strokes that are not words (underlines, dots). Its few parameters are tuned on
pages that neither the detector nor the benchmark was involved in.
"""

import math
from dataclasses import dataclass

import numpy as np

from xournalpp_htr.training.word_detector_online.data import StrokePage
from xournalpp_htr.training.word_detector_online.detectors import StrokeGrouper
from xournalpp_htr.training.word_detector_online.features import (
    bounding_boxes,
    page_scale,
)
from xournalpp_htr.training.word_detector_online.metrics import (
    GroupCounts,
    box_counts,
)


@dataclass(frozen=True)
class PostProcessParams:
    """All lengths are in units of the page's typical stroke size.

    The defaults leave the groups unchanged.
    """

    #: Groups whose box is at most this large (both sides) are fragments.
    fragment_size: float = 0.0
    #: A fragment merges into a neighbour on its line at most this far away.
    fragment_gap: float = 0.0
    #: A lone stroke at least this wide and this much wider than tall is an
    #: underline and dropped.
    underline_width: float = math.inf
    underline_aspect: float = math.inf
    #: A lone stroke at most this large (both sides) is a dot and dropped.
    dot_size: float = 0.0


def _union_box(boxes: np.ndarray, strokes: set[int]) -> np.ndarray:
    selected = boxes[sorted(strokes)]
    return np.array(
        [
            selected[:, 0].min(),
            selected[:, 1].min(),
            selected[:, 2].max(),
            selected[:, 3].max(),
        ]
    )


def _neighbour_gap(fragment: np.ndarray, other: np.ndarray) -> float | None:
    """Horizontal gap between two boxes on the same line, or None if not on one."""
    overlap = min(fragment[3], other[3]) - max(fragment[1], other[1])
    if overlap < 0.3 * (fragment[3] - fragment[1]):
        return None
    return float(max(fragment[0] - other[2], other[0] - fragment[2], 0.0))


def postprocess_groups(
    page: StrokePage, groups: list[frozenset[int]], params: PostProcessParams
) -> list[frozenset[int]]:
    """Merge fragments into neighbours, then drop underline and dot strokes.

    Dropped strokes belong to no returned group, so the result need not
    partition the strokes of the page.
    """
    stroke_boxes = bounding_boxes(page)
    scale = page_scale(stroke_boxes)
    members = [set(g) for g in groups]
    boxes = [_union_box(stroke_boxes, g) for g in members]

    def size(box: np.ndarray) -> float:
        return max(box[2] - box[0], box[3] - box[1]) / scale

    alive = set(range(len(members)))
    if params.fragment_size > 0.0:
        for i in sorted(alive, key=lambda k: size(boxes[k])):
            if size(boxes[i]) > params.fragment_size:
                break
            gaps = {
                j: gap
                for j in alive - {i}
                if (gap := _neighbour_gap(boxes[i], boxes[j])) is not None
                and gap <= params.fragment_gap * scale
            }
            if gaps:
                target = min(gaps, key=gaps.get)
                members[target] |= members[i]
                boxes[target] = _union_box(stroke_boxes, members[target])
                alive.discard(i)

    result = []
    for i in sorted(alive):
        if len(members[i]) == 1:
            width = (boxes[i][2] - boxes[i][0]) / scale
            height = (boxes[i][3] - boxes[i][1]) / scale
            is_underline = (
                width >= params.underline_width
                and width >= params.underline_aspect * max(height, 1e-6)
            )
            if is_underline or max(width, height) <= params.dot_size:
                continue
        result.append(frozenset(members[i]))
    return result


class PostProcessedGrouper(StrokeGrouper):
    """A grouper whose output goes through `postprocess_groups`."""

    def __init__(self, base: StrokeGrouper, params: PostProcessParams):
        super().__init__(base.beta)
        self.base = base
        self.params = params

    def fit(self, pages):
        self.base.fit(pages)
        return self

    def predict(self, page):
        return postprocess_groups(page, self.base.predict(page), self.params)


_FRAGMENT_SIZES = [0.0, 0.4, 0.6, 0.8, 1.0]
_FRAGMENT_GAPS = [0.3, 0.6, 1.0]
_UNDERLINE_WIDTHS = [2.0, 4.0]
_UNDERLINE_ASPECTS = [math.inf, 5.0, 8.0, 12.0]
_DOT_SIZES = [0.0, 0.15, 0.3]


def tune_postprocess(
    grouper: StrokeGrouper, pages: list[StrokePage], beta: float = 1.0
) -> PostProcessParams:
    """Choose the `PostProcessParams` scoring best on `pages` for a fitted grouper.

    Merging and dropping are tuned one after the other. The pages should not
    have been used to fit the grouper.
    """
    predictions = [grouper.predict(page) for page in pages]

    def score(params: PostProcessParams) -> float:
        total = GroupCounts()
        for page, groups in zip(pages, predictions, strict=True):
            total = total + box_counts(postprocess_groups(page, groups, params), page)
        return total.f_beta(beta)

    best, best_score = PostProcessParams(), score(PostProcessParams())
    for size in _FRAGMENT_SIZES[1:]:
        for gap in _FRAGMENT_GAPS:
            params = PostProcessParams(fragment_size=size, fragment_gap=gap)
            if (s := score(params)) > best_score:
                best, best_score = params, s

    merged = best
    for width in _UNDERLINE_WIDTHS:
        for aspect in _UNDERLINE_ASPECTS:
            for dot in _DOT_SIZES:
                params = PostProcessParams(
                    merged.fragment_size, merged.fragment_gap, width, aspect, dot
                )
                if (s := score(params)) > best_score:
                    best, best_score = params, s
    return best
