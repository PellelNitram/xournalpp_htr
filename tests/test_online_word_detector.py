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


def test_squeeze_word_gaps_shrinks_only_the_gaps():
    from xournalpp_htr.training.online_word_detector.iam_ondb import (
        _squeeze_word_gaps,
    )

    strokes = [
        (np.array([0.0, 1.0]), np.array([0.0, 0.0])),
        (np.array([5.0, 6.0]), np.array([0.0, 0.0])),
        (np.array([12.0, 13.0]), np.array([0.0, 0.0])),
    ]
    squeezed = _squeeze_word_gaps(strokes, [[0], [1], [2]], 0.5)
    # Gaps of 4 and 6 become 2 and 3; the words keep their width.
    assert [float(x.min()) for x, _ in squeezed] == [0.0, 3.0, 7.0]
    assert [float(x.max() - x.min()) for x, _ in squeezed] == [1.0, 1.0, 1.0]


def _line_page(boxes_and_strokes):
    """A page of one stroke per ``(xmin, xmax, ymin, ymax)``."""
    strokes = [
        (np.array([x0, x1]), np.array([y0, y1])) for x0, x1, y0, y1 in boxes_and_strokes
    ]
    return StrokePage("doc", 0, 0, strokes, np.full(len(strokes), NON_TEXT))


def test_postprocess_defaults_change_nothing():
    from xournalpp_htr.training.online_word_detector.postprocess import (
        PostProcessParams,
        postprocess_groups,
    )

    page = _page()
    groups = [frozenset(g) for g in page.groups]
    assert postprocess_groups(page, groups, PostProcessParams()) == groups


def test_postprocess_merges_fragment_and_drops_underline():
    from xournalpp_htr.training.online_word_detector.postprocess import (
        PostProcessParams,
        postprocess_groups,
    )

    # Two letters of one word, a detached small "s", and a long flat underline.
    page = _line_page(
        [(0, 4, 0, 10), (5, 9, 0, 10), (10, 12, 4, 10), (0, 40, 14, 14.5)]
    )
    groups = [frozenset({0, 1}), frozenset({2}), frozenset({3})]
    params = PostProcessParams(
        fragment_size=1.0, fragment_gap=0.5, underline_width=2.0, underline_aspect=5.0
    )
    assert postprocess_groups(page, groups, params) == [frozenset({0, 1, 2})]
