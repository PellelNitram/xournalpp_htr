"""Detectors that partition the strokes of a page into word groups.

Every detector is fit on training pages and then predicts, for a page, a
partition of all its strokes. Parameters of the rule-based detectors are tuned
on the training pages by grid search; the beta of the F-beta score they
maximise is a constructor argument (beta > 1 favours recall, which is what the
word detection step should optimise for).
"""

import itertools
from abc import ABC, abstractmethod

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import GradientBoostingClassifier

from xournalpp_htr.training.word_detector_online.data import NON_TEXT, StrokePage
from xournalpp_htr.training.word_detector_online.features import (
    PAIR_FEATURE_NAMES,
    box_distance_matrix,
    consecutive_pair_features,
)
from xournalpp_htr.training.word_detector_online.metrics import (
    GroupCounts,
    box_counts,
    group_counts,
)
from xournalpp_htr.training.word_detector_online.model import groups_from_merges

_GAP_X = PAIR_FEATURE_NAMES.index("gap_x")
_CENTER_DY = PAIR_FEATURE_NAMES.index("center_dy")


class StrokeGrouper(ABC):
    """Partitions the strokes of a page into groups, one group per word."""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    @abstractmethod
    def fit(self, pages: list[StrokePage]) -> "StrokeGrouper":
        """Learn or tune parameters on training pages."""

    @abstractmethod
    def predict(self, page: StrokePage) -> list[frozenset[int]]:
        """Return disjoint stroke index sets covering all strokes of the page."""

    def _score(self, pages: list[StrokePage]) -> float:
        total = GroupCounts()
        for page in pages:
            total = total + group_counts(self.predict(page), page)
        return total.f_beta(self.beta)

    def _grid_search(self, pages: list[StrokePage], grid: dict[str, list]) -> dict:
        """Set the parameters in `grid` to the combination scoring best on `pages`."""
        best_score, best_params = -1.0, {}
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), values, strict=False))
            for name, value in params.items():
                setattr(self, name, value)
            score = self._score(pages)
            if score > best_score:
                best_score, best_params = score, params
        for name, value in best_params.items():
            setattr(self, name, value)
        return best_params


class GapThresholdGrouper(StrokeGrouper):
    """Merge consecutive strokes whose boxes are close and on the same line."""

    def __init__(self, beta: float = 1.0):
        super().__init__(beta)
        self.max_gap = 0.5
        self.max_line_offset = 1.0

    def fit(self, pages):
        self._grid_search(
            pages,
            {
                "max_gap": [float(g) for g in np.arange(0.0, 3.01, 0.1)],
                "max_line_offset": [0.5, 1.0, 2.0, np.inf],
            },
        )
        return self

    def predict(self, page):
        features = consecutive_pair_features(page)
        merge = (features[:, _GAP_X] < self.max_gap) & (
            np.abs(features[:, _CENTER_DY]) < self.max_line_offset
        )
        return groups_from_merges(len(page.strokes), merge)


class ClusteringGrouper(StrokeGrouper):
    """Single-linkage clustering of stroke boxes, ignoring the writing order.

    Unlike the consecutive-stroke detectors it can reunite a word with strokes
    written later, such as the dot of an i or the bar of a t.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__(beta)
        self.max_distance = 0.5
        self.vertical_weight = 2.0

    def fit(self, pages):
        self._grid_search(
            pages,
            {
                "max_distance": [float(d) for d in np.arange(0.05, 3.01, 0.05)],
                "vertical_weight": [1.0, 2.0, 4.0],
            },
        )
        return self

    def predict(self, page):
        n = len(page.strokes)
        if n == 1:
            return [frozenset([0])]
        distances = box_distance_matrix(page, self.vertical_weight)
        np.fill_diagonal(distances, 0.0)
        tree = linkage(squareform(distances, checks=False), method="single")
        cluster_ids = fcluster(tree, t=self.max_distance, criterion="distance")
        return [
            frozenset(np.flatnonzero(cluster_ids == c).tolist())
            for c in np.unique(cluster_ids)
        ]


class PairwiseClassifierGrouper(StrokeGrouper):
    """Gradient boosting on features of consecutive strokes: same word or not.

    The label is "merge" iff both strokes belong to the same text group, so
    strokes of drawings and diagrams are split like word boundaries.
    """

    _features = staticmethod(consecutive_pair_features)

    def __init__(self, beta: float = 1.0, seed: int = 0):
        super().__init__(beta)
        self.seed = seed
        self.threshold = 0.5
        self._model = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, random_state=seed
        )

    def fit(self, pages):
        features = np.concatenate([self._features(p) for p in pages])
        labels = np.concatenate(
            [
                (p.labels[:-1] == p.labels[1:]) & (p.labels[:-1] != NON_TEXT)
                for p in pages
            ]
        )
        self._model.fit(features, labels)
        return self

    def _merge_probabilities(self, page: StrokePage) -> np.ndarray:
        if len(page.strokes) == 1:
            return np.empty(0)
        return self._model.predict_proba(self._features(page))[:, 1]

    def predict(self, page):
        return groups_from_merges(
            len(page.strokes), self._merge_probabilities(page) >= self.threshold
        )


class TunedPairwiseClassifierGrouper(PairwiseClassifierGrouper):
    """`PairwiseClassifierGrouper` with a merge threshold tuned on held-out documents.

    The classifier's probabilities are not calibrated, so 0.5 is a poor
    threshold. It is chosen on out-of-fold probabilities, obtained by leaving
    each training document out in turn, so no test page is involved.
    """

    MAX_TUNING_FOLDS = 6

    def fit(self, pages):
        super().fit(pages)
        documents = sorted({p.document for p in pages})
        if len(documents) < 2:
            return self

        # Hold out documents in at most `MAX_TUNING_FOLDS` groups, so that
        # training sets with very many documents (writers) stay affordable.
        n_folds = min(len(documents), self.MAX_TUNING_FOLDS)
        fold_of = {d: i % n_folds for i, d in enumerate(documents)}
        out_of_fold = []
        for fold in range(n_folds):
            inner = PairwiseClassifierGrouper(self.beta, self.seed).fit(
                [p for p in pages if fold_of[p.document] != fold]
            )
            out_of_fold += [
                (p, inner._merge_probabilities(p))
                for p in pages
                if fold_of[p.document] == fold
            ]

        best_score = -1.0
        for threshold in np.arange(0.05, 0.96, 0.05):
            total = GroupCounts()
            for page, proba in out_of_fold:
                groups = groups_from_merges(len(page.strokes), proba >= threshold)
                total = total + box_counts(groups, page)
            if total.f_beta(self.beta) > best_score:
                best_score = total.f_beta(self.beta)
                self.threshold = float(threshold)
        return self


GROUPERS: dict[str, type[StrokeGrouper]] = {
    "gap_threshold": GapThresholdGrouper,
    "clustering": ClusteringGrouper,
    "pairwise_gbm": PairwiseClassifierGrouper,
    "pairwise_gbm_tuned": TunedPairwiseClassifierGrouper,
}
