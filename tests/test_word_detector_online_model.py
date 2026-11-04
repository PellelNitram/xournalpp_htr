"""Tests for exporting and loading the WordDetectorOnline detector."""

import numpy as np
import pytest

from xournalpp_htr.inference_models import OnlineWordDetectorModel
from xournalpp_htr.training.word_detector_online.detectors import (
    PairwiseClassifierGrouper,
)
from xournalpp_htr.training.word_detector_online.export import write_artifacts
from xournalpp_htr.training.word_detector_online.features import (
    consecutive_pair_features,
)
from xournalpp_htr.training.word_detector_online.model import TreeEnsemble
from xournalpp_htr.training.word_detector_online.strokes import StrokePage


def _random_pages(n_pages: int = 20, seed: int = 0) -> list[StrokePage]:
    """Lines of words made of short strokes, with random spacing."""
    rng = np.random.default_rng(seed)
    pages = []
    for _ in range(n_pages):
        strokes, labels, x = [], [], 0.0
        for word in range(rng.integers(3, 9)):
            for _ in range(rng.integers(1, 6)):
                width, height = rng.uniform(0.3, 1.2), rng.uniform(0.6, 1.4)
                strokes.append((np.array([x, x + width]), np.array([0.0, height])))
                labels.append(word)
                x += width + rng.uniform(-0.1, 0.5)
            x += rng.uniform(1.0, 3.0)
        pages.append(StrokePage("doc", 0, 0, strokes, np.array(labels)))
    return pages


@pytest.fixture(scope="module")
def fitted():
    pages = _random_pages()
    return PairwiseClassifierGrouper().fit(pages), pages


def test_tree_ensemble_matches_scikit_learn(fitted):
    grouper, pages = fitted
    ensemble = TreeEnsemble.from_sklearn(grouper._model)
    for page in pages:
        features = consecutive_pair_features(page)
        np.testing.assert_allclose(
            ensemble.predict_proba(features),
            grouper._model.predict_proba(features)[:, 1],
            atol=1e-9,
        )


def test_tree_ensemble_save_load_roundtrip(fitted, tmp_path):
    grouper, pages = fitted
    ensemble = TreeEnsemble.from_sklearn(grouper._model)
    ensemble.save(tmp_path / "model.npz")
    loaded = TreeEnsemble.load(tmp_path / "model.npz")
    features = consecutive_pair_features(pages[0])
    np.testing.assert_array_equal(
        loaded.predict_proba(features), ensemble.predict_proba(features)
    )


def test_inference_model_groups_like_the_trained_grouper(fitted, tmp_path):
    grouper, pages = fitted
    grouper.threshold = 0.3  # not the default, so a wrong threshold is caught
    write_artifacts(grouper, tmp_path, training={"dataset": "synthetic"})
    model = OnlineWordDetectorModel.from_local(tmp_path)
    assert model.threshold == 0.3
    for page in pages:
        assert model.group_strokes(page.strokes) == grouper.predict(page)


def test_inference_model_handles_degenerate_input(fitted, tmp_path):
    grouper, pages = fitted
    write_artifacts(grouper, tmp_path, training={})
    model = OnlineWordDetectorModel.from_local(tmp_path)
    assert model.group_strokes([]) == []
    assert model.group_strokes(pages[0].strokes[:1]) == [frozenset([0])]
