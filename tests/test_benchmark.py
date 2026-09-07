from dataclasses import dataclass

import pytest

from xournalpp_htr.benchmark import (
    BenchmarkResult,
    GroundTruthWord,
    PageDetail,
    ReportData,
    SampleDetail,
    WordRow,
    _cer,
    _collect_pages,
    _iou,
    _match,
    _matches,
    write_html_report,
)
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


# --- _matches ---


def test_matches_identical_boxes():
    assert _matches(gt(), pred())


def test_matches_padded_prediction_around_short_word():
    # The case IoU could not handle: a two letter word whose box is small
    # enough that the detector's padding dominates. IoU here is only ~0.23.
    g = gt(text="is", xmin=0, xmax=12, ymin=0, ymax=10)
    p = pred(text="is", xmin=-6, xmax=18, ymin=-6, ymax=16)
    assert _iou(g, p) < 0.3
    assert _matches(g, p)


def test_matches_rejects_partial_coverage():
    # Prediction covers only half of the GT box.
    g = gt(xmin=0, xmax=20, ymin=0, ymax=10)
    p = pred(xmin=10, xmax=30, ymin=0, ymax=10)
    assert not _matches(g, p)


def test_matches_rejects_oversized_box():
    # Fully covered, but the prediction is far larger than the word -- e.g. a
    # box spanning a whole line.
    g = gt(xmin=0, xmax=10, ymin=0, ymax=10)
    p = pred(xmin=-5, xmax=400, ymin=-5, ymax=25)
    assert not _matches(g, p)


def test_matches_accepts_at_coverage_boundary():
    # Exactly 80% of the GT box is covered.
    g = gt(xmin=0, xmax=10, ymin=0, ymax=10)
    p = pred(xmin=0, xmax=8, ymin=0, ymax=10)
    assert _matches(g, p)


def test_matches_rejects_just_below_coverage_boundary():
    # 79% covered -- one percent short of the boundary above.
    g = gt(xmin=0, xmax=100, ymin=0, ymax=10)
    p = pred(xmin=0, xmax=79, ymin=0, ymax=10)
    assert not _matches(g, p)


def test_matches_rejects_prediction_smaller_than_word():
    # A box sitting inside the word, e.g. the detector split it in half. The
    # criterion is one-directional: covering the prediction is not enough,
    # the prediction has to cover the word.
    g = gt(xmin=0, xmax=40, ymin=0, ymax=10)
    p = pred(xmin=0, xmax=15, ymin=0, ymax=10)
    assert not _matches(g, p)


def test_matches_zero_area_ground_truth():
    # A single point annotation must not blow up the ratio computations.
    g = gt(xmin=5, xmax=5, ymin=5, ymax=5)
    assert not _matches(g, pred())


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
    # Both predictions cover the word, so both are acceptable matches; the
    # tighter one has to win on IoU.
    g = gt(xmin=0, xmax=10, ymin=0, ymax=10)
    p_close = pred(xmin=0, xmax=10, ymin=0, ymax=10)  # iou=1.0
    p_padded = pred(xmin=-4, xmax=14, ymin=0, ymax=10)  # iou≈0.56
    assert _matches(g, p_close) and _matches(g, p_padded)

    pairs = _match([g], {0: [p_close, p_padded]})
    assert len(pairs) == 1
    assert pairs[0][1] is p_close


def test_match_one_pred_two_gts_assigns_higher_iou():
    # The prediction is an acceptable match for both words; it has to go to
    # the one it fits tightest.
    g_close = gt(xmin=0, xmax=10, ymin=0, ymax=10)
    g_inside = gt(xmin=2, xmax=9, ymin=2, ymax=9)
    p = pred(xmin=0, xmax=10, ymin=0, ymax=10)  # iou=1.0 with g_close
    assert _matches(g_close, p) and _matches(g_inside, p)

    pairs = _match([g_close, g_inside], {0: [p]})
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


# --- report details ---


@dataclass
class FakePage:
    meta_data: dict
    layers: list


class FakeDocument:
    """Stand-in for `Document` that renders a fixed byte string per page."""

    def __init__(self, n_pages=1):
        self.pages = [
            FakePage(meta_data={"width": "595", "height": "842"}, layers=[])
            for _ in range(n_pages)
        ]

    def save_page_as_image(self, page_index, out_path, black_white=False, dpi=72.0):
        out_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return out_path


def test_collect_pages_classifies_words():
    g_correct = gt(text="hello", xmin=0, xmax=10, ymin=0, ymax=10)
    g_wrong = gt(text="world", xmin=20, xmax=30, ymin=0, ymax=10)
    g_missed = gt(text="miss", xmin=100, xmax=110, ymin=100, ymax=110)
    p_correct = pred(text="hello", xmin=0, xmax=10, ymin=0, ymax=10)
    p_wrong = pred(text="wor1d", xmin=20, xmax=30, ymin=0, ymax=10)
    p_spurious = pred(text="ghost", xmin=200, xmax=210, ymin=200, ymax=210)

    gt_words = [g_correct, g_wrong, g_missed]
    predictions = {0: [p_correct, p_wrong, p_spurious]}
    pages = _collect_pages(
        FakeDocument(), gt_words, predictions, _match(gt_words, predictions)
    )

    assert len(pages) == 1
    page = pages[0]
    assert page.width == 595.0
    assert page.height == 842.0
    assert page.image_uri.startswith("data:image/png;base64,")

    by_status = {row.status: row for row in page.rows}
    assert set(by_status) == {"correct", "wrong", "missed", "spurious"}
    assert by_status["correct"].pred_text == "hello"
    assert by_status["wrong"].gt_text == "world"
    assert by_status["wrong"].pred_text == "wor1d"
    assert by_status["missed"].pred_text is None
    assert by_status["missed"].pred_box is None
    assert by_status["spurious"].gt_text is None
    assert by_status["spurious"].gt_box is None


def test_collect_pages_status_is_case_sensitive():
    g = gt(text="Hello")
    p = pred(text="hello")
    pages = _collect_pages(FakeDocument(), [g], {0: [p]}, _match([g], {0: [p]}))
    assert pages[0].rows[0].status == "wrong"


def test_collect_pages_distinguishes_equal_predictions():
    # `WordPrediction` is a dataclass, so two identical predictions compare
    # equal. Only one of them can be matched; the other has to come back as
    # spurious rather than being conflated with it.
    g = gt(text="hello")
    p_first = pred(text="hello")
    p_second = pred(text="hello")
    assert p_first == p_second

    predictions = {0: [p_first, p_second]}
    pairs = _match([g], predictions)
    assert len(pairs) == 1

    rows = _collect_pages(FakeDocument(), [g], predictions, pairs)[0].rows
    assert sorted(row.status for row in rows) == ["correct", "spurious"]


def test_collect_pages_covers_pages_without_predictions():
    g = gt(text="alone", page_index=1)
    pages = _collect_pages(FakeDocument(n_pages=2), [g], {}, [])
    assert [len(page.rows) for page in pages] == [0, 1]
    assert pages[1].rows[0].status == "missed"


# --- write_html_report ---


def report_data(rows):
    page = PageDetail(
        page_index=0,
        width=595.0,
        height=842.0,
        image_uri="data:image/png;base64,iVBORw0KGgo=",
        rows=rows,
    )
    details = ReportData(
        pipeline_name="test_pipeline",
        git_sha="abc1234",
        created_at="2026-08-08 12:00:00",
    )
    details.samples.append(SampleDetail(name="sample.xopp", pages=[page]))
    return BenchmarkResult(0.5, 0.5, 0.1, 0.1, 0.45, 0.5, 4, 4, 2, details=details)


def test_write_html_report_without_details_raises(tmp_path):
    result = BenchmarkResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)
    with pytest.raises(ValueError):
        write_html_report(result, tmp_path / "report.html")


def test_write_html_report_writes_self_contained_page(tmp_path):
    rows = [
        WordRow("correct", "hello", "hello", (0, 0, 10, 10), (0, 0, 10, 10)),
        WordRow("wrong", "world", "wor1d", (20, 0, 30, 10), (20, 0, 30, 10)),
        WordRow("missed", "miss", None, (40, 0, 50, 10), None),
        WordRow("spurious", None, "ghost", None, (60, 0, 70, 10)),
    ]
    out = tmp_path / "nested" / "report.html"
    assert write_html_report(report_data(rows), out) == out

    content = out.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "test_pipeline" in content
    assert "abc1234" in content
    assert "sample.xopp" in content
    assert "data:image/png;base64," in content
    # Predictions are drawn, the unfound word is underlined instead.
    assert "wor1d" in content
    assert "<line" in content
    # Nothing is loaded from outside the file.
    assert "http://" not in content
    assert "https://" not in content


def test_write_html_report_escapes_text(tmp_path):
    rows = [WordRow("wrong", "a<b>", "x&y", (0, 0, 10, 10), (0, 0, 10, 10))]
    out = tmp_path / "report.html"
    write_html_report(report_data(rows), out)

    content = out.read_text(encoding="utf-8")
    assert "&lt;b&gt;" in content
    assert "x&amp;y" in content
    assert "<b>" not in content
