"""Numpy-only evaluation of the trained pairwise gradient boosting model.

Training uses scikit-learn; the trained trees are stored as plain arrays so the
inference install needs neither scikit-learn nor pickle, and a saved model does
not depend on a scikit-learn version.
"""

from pathlib import Path

import numpy as np


def groups_from_merges(
    n_strokes: int, merge_with_next: np.ndarray
) -> list[frozenset[int]]:
    """Cut the stroke sequence wherever ``merge_with_next[i]`` is False."""
    groups, current = [], [0]
    for i in range(1, n_strokes):
        if merge_with_next[i - 1]:
            current.append(i)
        else:
            groups.append(frozenset(current))
            current = [i]
    groups.append(frozenset(current))
    return groups


class TreeEnsemble:
    """Gradient boosted regression trees of a binary classifier, as arrays.

    All trees are concatenated; ``roots[t]`` is the node index of tree ``t``'s
    root, child indices are absolute and ``-1`` marks a leaf.
    """

    ARRAYS = ("roots", "left", "right", "feature", "threshold", "value")

    def __init__(
        self,
        init_raw: float,
        learning_rate: float,
        roots: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        feature: np.ndarray,
        threshold: np.ndarray,
        value: np.ndarray,
    ):
        self.init_raw = float(init_raw)
        self.learning_rate = float(learning_rate)
        self.roots = roots
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value

    @classmethod
    def from_sklearn(cls, model) -> "TreeEnsemble":
        """Extract the trees of a fitted binary ``GradientBoostingClassifier``."""
        roots, left, right, feature, threshold, value = [], [], [], [], [], []
        for (estimator,) in model.estimators_:
            tree, offset = estimator.tree_, len(left)
            roots.append(offset)
            left.extend(
                np.where(tree.children_left < 0, -1, tree.children_left + offset)
            )
            right.extend(
                np.where(tree.children_right < 0, -1, tree.children_right + offset)
            )
            feature.extend(tree.feature)
            threshold.extend(tree.threshold)
            value.extend(tree.value[:, 0, 0])
        ensemble = cls(
            0.0,
            model.learning_rate,
            np.array(roots),
            np.array(left),
            np.array(right),
            np.array(feature),
            np.array(threshold, dtype=np.float64),
            np.array(value, dtype=np.float64),
        )
        # The starting score is whatever the model adds to the trees' sum; read
        # it off one row instead of reproducing scikit-learn's initialisation.
        probe = np.zeros((1, model.n_features_in_))
        ensemble.init_raw = float(
            model.decision_function(probe)[0] - ensemble._tree_sum(probe)[0]
        )
        return ensemble

    def _tree_sum(self, features: np.ndarray) -> np.ndarray:
        """Learning-rate-scaled sum of the leaf values reached in every tree."""
        # scikit-learn compares float32 features against the float64 thresholds.
        x = np.asarray(features, dtype=np.float32)
        rows = np.arange(len(x))
        total = np.zeros(len(x))
        for root in self.roots:
            node = np.full(len(x), root)
            while True:
                internal = self.left[node] >= 0
                if not internal.any():
                    break
                go_left = x[rows, self.feature[node]] <= self.threshold[node]
                step = np.where(go_left, self.left[node], self.right[node])
                node = np.where(internal, step, node)
            total += self.value[node]
        return self.learning_rate * total

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probability of the positive class for each row of `features`."""
        raw = self.init_raw + self._tree_sum(features)
        return 1.0 / (1.0 + np.exp(-raw))

    def save(self, path: Path) -> None:
        np.savez(
            path,
            init_raw=self.init_raw,
            learning_rate=self.learning_rate,
            **{name: getattr(self, name) for name in self.ARRAYS},
        )

    @classmethod
    def load(cls, path: Path) -> "TreeEnsemble":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                float(data["init_raw"]),
                float(data["learning_rate"]),
                *(data[name] for name in cls.ARRAYS),
            )
