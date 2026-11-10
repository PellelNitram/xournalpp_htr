"""Snapping predicted word boxes to the pen strokes inside them (issue #171)."""

import numpy as np

from xournalpp_htr.models import WordPrediction, snap_predictions_to_strokes


def _stroke(x0, y0, x1, y1):
    return np.array([x0, x1], dtype=float), np.array([y0, y1], dtype=float)


def _box(text, xmin, ymin, xmax, ymax):
    return WordPrediction(text=text, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def test_box_becomes_the_bounds_of_its_strokes():
    predictions = [_box("hi", 0, 0, 100, 50)]
    strokes = [_stroke(10, 10, 30, 20), _stroke(40, 12, 60, 35)]

    (snapped,) = snap_predictions_to_strokes(predictions, strokes)

    assert snapped.text == "hi"
    assert (snapped.xmin, snapped.ymin, snapped.xmax, snapped.ymax) == (10, 10, 60, 35)


def test_stroke_goes_to_the_box_holding_most_of_it():
    predictions = [_box("a", 0, 0, 50, 50), _box("b", 50, 0, 100, 50)]
    # Three quarters of its points inside "b".
    stroke = (np.array([40.0, 60.0, 70.0, 80.0]), np.full(4, 25.0))

    first, second = snap_predictions_to_strokes(predictions, [stroke])

    assert (first.xmin, first.xmax) == (0, 50)  # got no stroke: unchanged
    assert (second.xmin, second.xmax) == (40, 80)


def test_ties_go_to_the_smaller_box():
    predictions = [_box("wide", 0, 0, 200, 50), _box("tight", 10, 10, 40, 30)]
    stroke = _stroke(15, 15, 30, 25)

    wide, tight = snap_predictions_to_strokes(predictions, [stroke])

    assert (wide.xmin, wide.xmax) == (0, 200)
    assert (tight.xmin, tight.xmax) == (15, 30)


def test_stroke_mostly_outside_every_box_is_ignored():
    predictions = [_box("a", 0, 0, 50, 50)]
    stroke = (np.array([40.0, 120.0, 130.0, 140.0]), np.full(4, 25.0))

    (snapped,) = snap_predictions_to_strokes(predictions, [stroke])

    assert (snapped.xmin, snapped.xmax) == (0, 50)


def test_nothing_to_snap_returns_the_predictions():
    prediction = _box("a", 0, 0, 50, 50)
    assert snap_predictions_to_strokes([prediction], []) == [prediction]
    assert snap_predictions_to_strokes([], [_stroke(0, 0, 1, 1)]) == []
