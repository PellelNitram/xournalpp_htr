"""Tests for the stroke-based word detectors (issue #168)."""

import numpy as np
import pytest

from xournalpp_htr.training.online_word_detector.data import NON_TEXT, StrokePage
from xournalpp_htr.training.online_word_detector.detectors import GROUPERS
from xournalpp_htr.training.online_word_detector.synthetic import (
    KINDS,
    random_element,
)


def _page(n_words: int = 4, strokes_per_word: int = 3) -> StrokePage:
    """A line of words made of short vertical strokes, with wide word gaps."""
    strokes, labels = [], []
    for word in range(n_words):
        for k in range(strokes_per_word):
            x = word * 10.0 + k * 0.6
            strokes.append((np.array([x, x + 0.1]), np.array([0.0, 1.0])))
            labels.append(word)
    return StrokePage("doc", 0, 0, strokes, np.array(labels))


@pytest.mark.parametrize("kind", KINDS)
def test_synthetic_element_has_finite_strokes(kind):
    rng = np.random.default_rng(0)
    strokes = random_element(kind, (0.0, 0.0, 100.0, 10.0), 5.0, rng)
    assert strokes
    for x, y in strokes:
        assert len(x) == len(y) >= 2
        assert np.isfinite(x).all() and np.isfinite(y).all()


@pytest.mark.parametrize("name", list(GROUPERS))
def test_grouper_partitions_all_strokes(name):
    pages = [_page(), _page(5, 2)]
    pages[1].document = "other"
    grouper = GROUPERS[name]().fit(pages)
    groups = grouper.predict(_page())
    assert sorted(i for g in groups for i in g) == list(range(12))


def test_gap_threshold_recovers_clear_words():
    page = _page()
    groups = GROUPERS["gap_threshold"]().fit([page]).predict(page)
    assert set(groups) == set(page.groups)


def test_groups_ignore_non_text():
    page = _page()
    page.labels[0] = NON_TEXT
    assert all(0 not in g for g in page.groups)
