"""Fast, pure unit tests for the shared WordDetector post-processing."""

import numpy as np
import pytest

from xournalpp_htr.training.shared.bounding_box import BoundingBox, ImageDimensions
from xournalpp_htr.training.shared.postprocessing import (
    MapOrdering,
    binary_classification_metrics,
    cluster_aabbs,
    compute_iou,
    compute_scale_down,
    decode,
    normalize_image_transform,
)

# --- BoundingBox ---


def test_bounding_box_translate_and_scale():
    b = BoundingBox(1, 2, 3, 4, label="w")
    t = b.translate(10, 20)
    assert (t.x_min, t.y_min, t.x_max, t.y_max) == (11, 22, 13, 24)
    assert t.label == "w"
    s = b.scale(2, 3)
    assert (s.x_min, s.y_min, s.x_max, s.y_max) == (2, 6, 6, 12)


def test_bounding_box_area_and_clip():
    assert BoundingBox(0, 0, 10, 5).area() == 50.0
    clipped = BoundingBox(-5, -5, 100, 100).clip(BoundingBox(0, 0, 20, 20))
    assert (clipped.x_min, clipped.y_min, clipped.x_max, clipped.y_max) == (
        0,
        0,
        20,
        20,
    )


def test_scale_around_center_keeps_center():
    b = BoundingBox(0, 0, 10, 10)
    s = b.scale_around_center(0.5, 0.5)
    assert ((s.x_min + s.x_max) / 2, (s.y_min + s.y_max) / 2) == (5.0, 5.0)
    assert (s.x_min, s.x_max) == (2.5, 7.5)


def test_image_dimensions_namedtuple():
    d = ImageDimensions(height=224, width=448)
    assert d.height == 224 and d.width == 448


# --- iou / metrics ---


def test_compute_iou_identical_and_disjoint():
    a = BoundingBox(0, 0, 10, 10)
    assert compute_iou(a, BoundingBox(0, 0, 10, 10)) == pytest.approx(1.0)
    assert compute_iou(a, BoundingBox(20, 20, 30, 30)) == 0


def test_binary_classification_metrics_perfect_match():
    boxes = [BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)]
    m = binary_classification_metrics(boxes, boxes)
    assert m == {"tp": 2, "fp": 0, "fn": 0}


def test_compute_scale_down():
    assert compute_scale_down((448, 448), (224, 224)) == 0.5


# --- normalisation ---


def test_normalize_image_transform_range():
    img = np.array([[0, 255]], dtype=np.float32)
    out, bb = normalize_image_transform(img, "passthrough")
    assert out.min() == pytest.approx(-0.5)
    assert out.max() == pytest.approx(0.5)
    assert bb == "passthrough"


# --- decode ---


def test_decode_recovers_box_from_geometry_maps():
    h = w = 16
    pred = np.zeros((MapOrdering.NUM_MAPS, h, w), dtype=np.float32)
    yc, xc = 8, 8
    pred[MapOrdering.SEG_WORD, yc, xc] = 1.0
    pred[MapOrdering.GEO_TOP, yc, xc] = 2.0
    pred[MapOrdering.GEO_BOTTOM, yc, xc] = 3.0
    pred[MapOrdering.GEO_LEFT, yc, xc] = 4.0
    pred[MapOrdering.GEO_RIGHT, yc, xc] = 5.0

    boxes = decode(pred)
    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x_min, b.y_min, b.x_max, b.y_max) == (xc - 4, yc - 2, xc + 5, yc + 3)


def test_decode_empty_when_no_foreground():
    pred = np.zeros((MapOrdering.NUM_MAPS, 8, 8), dtype=np.float32)
    assert decode(pred) == []


# --- clustering ---


def test_cluster_aabbs_passthrough_for_small_input():
    boxes = [BoundingBox(0, 0, 10, 10)]
    assert cluster_aabbs(boxes) == boxes


def test_cluster_aabbs_merges_overlapping_group():
    # Five near-identical boxes (>= DBSCAN min_samples) collapse to one.
    boxes = [BoundingBox(0, 0, 10, 10).translate(i * 0.1, i * 0.1) for i in range(5)]
    clustered = cluster_aabbs(boxes)
    assert len(clustered) == 1
    assert clustered[0].x_min == pytest.approx(0.2, abs=0.21)
