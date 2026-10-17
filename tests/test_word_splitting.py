import numpy as np
import pytest

from xournalpp_htr.training.simple_htr_ondb.word_splitting import (
    group_tokens,
    split_into_words,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A MOVE to stop", ["A", "MOVE", "to", "stop"]),
        ("Mr . Gaitskell", ["Mr.", "Gaitskell"]),
        ("tomorrow , said ( maybe )", ["tomorrow,", "said", "(maybe)"]),
    ],
)
def test_group_tokens_glues_punctuation(text, expected):
    words, reason = group_tokens(text)
    assert reason is None
    assert words == expected


@pytest.mark.parametrize(
    "text, reason",
    [
        ('he said " hello', "ambiguous_punctuation"),
        (". hello", "leading_punctuation"),
        ("hello (", "dangling_opening_punctuation"),
        ("", "empty_line"),
    ],
)
def test_group_tokens_rejects_ambiguous_lines(text, reason):
    words, got = group_tokens(text)
    assert words == []
    assert got == reason


def test_split_cuts_at_largest_gaps():
    # Three words; the i-dot stroke (index 3) overlaps the first word.
    extents = np.array([[0, 10], [12, 20], [40, 55], [5, 6], [80, 95]], dtype=float)
    groups, reason = split_into_words(extents, n_words=3)
    assert reason is None
    assert [sorted(g) for g in groups] == [[0, 1, 3], [2], [4]]


def test_split_rejects_too_few_blobs():
    extents = np.array([[0, 10], [5, 15]], dtype=float)
    groups, reason = split_into_words(extents, n_words=2)
    assert groups is None
    assert reason == "too_few_blobs"


def test_split_rejects_ambiguous_gaps():
    # Gaps 10, 9, 9: two words need one cut but 10 is not clearly the largest.
    extents = np.array([[0, 5], [15, 20], [29, 34], [43, 48]], dtype=float)
    groups, reason = split_into_words(extents, n_words=2)
    assert groups is None
    assert reason == "ambiguous_gaps"


def test_split_single_word_keeps_everything():
    extents = np.array([[0, 5], [7, 9]], dtype=float)
    groups, _ = split_into_words(extents, n_words=1)
    assert [sorted(g) for g in groups] == [[0, 1]]
