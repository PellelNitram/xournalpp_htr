"""Tests for the stroke-based word detector (issue #168)."""

import numpy as np
import pytest

from xournalpp_htr.training.word_detector_online.data import NON_TEXT, StrokePage
from xournalpp_htr.training.word_detector_online.detectors import GROUPERS


def _page(n_words: int = 4, strokes_per_word: int = 3) -> StrokePage:
    """A line of words made of short vertical strokes, with wide word gaps."""
    strokes, labels = [], []
    for word in range(n_words):
        for k in range(strokes_per_word):
            x = word * 10.0 + k * 0.6
            strokes.append((np.array([x, x + 0.1]), np.array([0.0, 1.0])))
            labels.append(word)
    return StrokePage("doc", 0, 0, strokes, np.array(labels))


@pytest.mark.parametrize("name", list(GROUPERS))
def test_grouper_partitions_all_strokes(name):
    pages = [_page(), _page(5, 2)]
    pages[1].document = "other"
    grouper = GROUPERS[name]().fit(pages)
    groups = grouper.predict(_page())
    assert sorted(i for g in groups for i in g) == list(range(12))


def test_groups_ignore_non_text():
    page = _page()
    page.labels[0] = NON_TEXT
    assert all(0 not in g for g in page.groups)


def test_squeeze_word_gaps_shrinks_only_the_gaps():
    from xournalpp_htr.training.word_detector_online.iam_ondb import (
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
