import pytest

from xournalpp_htr.benchmark import GroundTruthWord, _cer, _iou, _match
from xournalpp_htr.models import WordPrediction


def gt(text="word", xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, page_index=0):
    return GroundTruthWord(
        text=text, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, page_index=page_index
    )


def pred(text="word", xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0):
    return WordPrediction(text=text, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


# --- _iou ---


def test_iou_identical_boxes():
    assert _iou(gt(), pred()) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert _iou(gt(xmin=0, xmax=10), pred(xmin=20, xmax=30)) == 0.0


def test_iou_partial_overlap():
    # gt: [0,10], pred: [5,15] — overlap is [5,10] = 5 wide
    a = gt(xmin=0, xmax=10, ymin=0, ymax=10)
    b = pred(xmin=5, xmax=15, ymin=0, ymax=10)
    # intersection = 5*10 = 50, union = 150
    assert _iou(a, b) == pytest.approx(50 / 150)


# --- _cer ---


def test_cer_identical():
    assert _cer("hello", "hello") == pytest.approx(0.0)


def test_cer_one_substitution():
    # "hello" vs "hella" — 1 substitution, 5 chars
    assert _cer("hello", "hella") == pytest.approx(1 / 5)


def test_cer_completely_wrong():
    assert _cer("abc", "xyz") == pytest.approx(1.0)


def test_cer_empty_hypothesis():
    # 3 deletions out of 3 chars
    assert _cer("abc", "") == pytest.approx(1.0)


# --- _match ---


def test_match_perfect_overlap():
    g = gt(text="hello")
    p = pred(text="helo")
    pairs = _match([g], {0: [p]})
    assert len(pairs) == 1
    assert pairs[0] == (g, p)


def test_match_no_overlap():
    g = gt(xmin=0, xmax=10, ymin=0, ymax=10)
    p = pred(xmin=50, xmax=60, ymin=50, ymax=60)
    assert _match([g], {0: [p]}) == []


def test_match_one_gt_two_preds_assigns_higher_iou():
    g = gt(xmin=0, xmax=10, ymin=0, ymax=10)
    p_close = pred(xmin=0, xmax=10, ymin=0, ymax=10)  # iou=1.0
    p_far = pred(xmin=5, xmax=15, ymin=0, ymax=10)  # iou<1.0
    pairs = _match([g], {0: [p_close, p_far]})
    assert len(pairs) == 1
    assert pairs[0][1] is p_close


def test_match_one_pred_two_gts_assigns_higher_iou():
    g_close = gt(xmin=0, xmax=10, ymin=0, ymax=10)
    g_far = gt(xmin=5, xmax=15, ymin=0, ymax=10)
    p = pred(xmin=0, xmax=10, ymin=0, ymax=10)  # iou=1.0 with g_close
    pairs = _match([g_close, g_far], {0: [p]})
    assert len(pairs) == 1
    assert pairs[0][0] is g_close


def test_match_respects_page_index():
    g = gt(page_index=0)
    p = pred()
    # prediction is on page 1, GT is on page 0 — no match expected
    assert _match([g], {1: [p]}) == []


def test_match_multiple_pages():
    g0 = gt(text="a", page_index=0)
    g1 = gt(text="b", page_index=1)
    p0 = pred(text="a")
    p1 = pred(text="b")
    pairs = _match([g0, g1], {0: [p0], 1: [p1]})
    assert len(pairs) == 2
