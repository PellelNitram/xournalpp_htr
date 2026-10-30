"""Scores of a stroke grouping against the ground truth text groups."""

from dataclasses import dataclass

import numpy as np

from xournalpp_htr.benchmark import GroundTruthWord, _match
from xournalpp_htr.models import WordPrediction
from xournalpp_htr.training.online_word_detector.data import StrokePage


@dataclass
class GroupCounts:
    """Exact-match counts: a predicted group is correct iff it is a GT group."""

    n_correct: int = 0
    n_predicted: int = 0
    n_gt: int = 0

    def __add__(self, other: "GroupCounts") -> "GroupCounts":
        return GroupCounts(
            self.n_correct + other.n_correct,
            self.n_predicted + other.n_predicted,
            self.n_gt + other.n_gt,
        )

    @property
    def precision(self) -> float:
        return self.n_correct / self.n_predicted if self.n_predicted else 0.0

    @property
    def recall(self) -> float:
        return self.n_correct / self.n_gt if self.n_gt else 0.0

    def f_beta(self, beta: float = 1.0) -> float:
        p, r = self.precision, self.recall
        if p == 0.0 and r == 0.0:
            return 0.0
        return (1 + beta**2) * p * r / (beta**2 * p + r)


def group_counts(predicted: list[frozenset[int]], page: StrokePage) -> GroupCounts:
    """Count the predicted groups that are exactly a ground truth text group.

    Predicted groups made of non-text strokes count as wrong, as a box around
    them would be a spurious detection.
    """
    gt = set(page.groups)
    return GroupCounts(
        n_correct=sum(group in gt for group in predicted),
        n_predicted=len(predicted),
        n_gt=len(gt),
    )


def _box(page: StrokePage, strokes: frozenset[int]) -> tuple[float, ...]:
    xs = np.concatenate([page.strokes[i][0] for i in strokes])
    ys = np.concatenate([page.strokes[i][1] for i in strokes])
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def box_counts(predicted: list[frozenset[int]], page: StrokePage) -> GroupCounts:
    """Count the predicted groups whose box the benchmark accepts as a GT word.

    Uses the benchmark's own matching (coverage of the GT box, capped area), so
    the result is comparable to the benchmark's detection precision and recall.
    """
    gt_words = [
        GroundTruthWord("", *_box(page, group), page_index=page.page_index)
        for group in page.groups
    ]
    predictions = {
        page.page_index: [WordPrediction("", *_box(page, group)) for group in predicted]
    }
    return GroupCounts(
        n_correct=len(_match(gt_words, predictions)),
        n_predicted=len(predicted),
        n_gt=len(gt_words),
    )
