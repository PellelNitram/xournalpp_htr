"""WordDetector map decoding, clustering and evaluation post-processing.

Base-deps-only (numpy, opencv, scikit-learn). Shared between training-time
evaluation and inference-time decoding of the network's segmentation/geometry
maps into word bounding boxes. See ADR 006 section 3.
"""

from collections import defaultdict
from typing import List

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from xournalpp_htr.training.shared.bounding_box import BoundingBox


class MapOrdering:
    """Channel order of the maps encoding the AABBs around the words."""

    SEG_WORD = 0
    SEG_SURROUNDING = 1
    SEG_BACKGROUND = 2
    GEO_TOP = 3
    GEO_BOTTOM = 4
    GEO_LEFT = 5
    GEO_RIGHT = 6
    NUM_MAPS = 7


def compute_scale_down(input_size, output_size) -> float:
    """Compute the network scale-down factor from input and output size."""
    return output_size[0] / input_size[0]


def subsample(idx, max_num):
    """Restrict foreground indices to a maximum number."""
    f = len(idx[0]) / max_num
    if f > 1:
        a = np.asarray([idx[0][int(j * f)] for j in range(max_num)], np.int64)
        b = np.asarray([idx[1][int(j * f)] for j in range(max_num)], np.int64)
        idx = (a, b)
    return idx


def fg_by_threshold(thres, max_num=None):
    """All pixels above threshold are fg pixels, optionally capped."""

    def func(seg_map):
        idx = np.where(seg_map > thres)
        if max_num is not None:
            idx = subsample(idx, max_num)
        return idx

    return func


def fg_by_cc(thres, max_num):
    """Take a max number of pixels per connected component (>=3 for DBSCAN)."""

    def func(seg_map):
        seg_mask = (seg_map > thres).astype(np.uint8)
        num_labels, label_img = cv2.connectedComponents(seg_mask, connectivity=4)
        max_num_per_cc = max(max_num // (num_labels + 1), 3)

        all_idx = [np.empty(0, np.int64), np.empty(0, np.int64)]
        for curr_label in range(1, num_labels):
            curr_idx = np.where(label_img == curr_label)
            curr_idx = subsample(curr_idx, max_num_per_cc)
            all_idx[0] = np.append(all_idx[0], curr_idx[0])
            all_idx[1] = np.append(all_idx[1], curr_idx[1])
        return tuple(all_idx)

    return func


def decode(
    nn_prediction,
    scale=1.0,
    comp_fg=fg_by_threshold(0.5),  # noqa: B008
) -> List[BoundingBox]:
    idx = comp_fg(nn_prediction[MapOrdering.SEG_WORD])
    nn_prediction_masked = nn_prediction[..., idx[0], idx[1]]
    bounding_boxes = []
    for yc, xc, pred in zip(idx[0], idx[1], nn_prediction_masked.T, strict=True):
        t = pred[MapOrdering.GEO_TOP]
        b = pred[MapOrdering.GEO_BOTTOM]
        l = pred[MapOrdering.GEO_LEFT]  # noqa: E741
        r = pred[MapOrdering.GEO_RIGHT]
        bbox = BoundingBox(x_min=xc - l, x_max=xc + r, y_min=yc - t, y_max=yc + b)
        bounding_boxes.append(bbox.scale(scale, scale))
    return bounding_boxes


def compute_iou(ra: BoundingBox, rb: BoundingBox) -> float:
    """Intersection over union of two axis-aligned rectangles."""
    if (
        ra.x_max < rb.x_min
        or rb.x_max < ra.x_min
        or ra.y_max < rb.y_min
        or rb.y_max < ra.y_min
    ):
        return 0

    l = max(ra.x_min, rb.x_min)  # noqa: E741
    r = min(ra.x_max, rb.x_max)
    t = max(ra.y_min, rb.y_min)
    b = min(ra.y_max, rb.y_max)

    intersection = (r - l) * (b - t)
    union = ra.area() + rb.area() - intersection
    return intersection / union


def compute_dist_mat(aabbs: List[BoundingBox]) -> np.ndarray:
    """Jaccard distance matrix of all pairs of aabbs."""
    num_aabbs = len(aabbs)
    dists = np.zeros((num_aabbs, num_aabbs))
    for i in range(num_aabbs):
        for j in range(num_aabbs):
            if j > i:
                break
            dists[i, j] = dists[j, i] = 1 - compute_iou(aabbs[i], aabbs[j])
    return dists


def cluster_aabbs(aabbs: List[BoundingBox]) -> List[BoundingBox]:
    """Cluster aabbs with DBSCAN on the Jaccard distance between boxes."""
    if len(aabbs) < 2:
        return aabbs

    dists = compute_dist_mat(aabbs)
    clustering = DBSCAN(eps=0.7, min_samples=3, metric="precomputed").fit(dists)

    clusters = defaultdict(list)
    for i, c in enumerate(clustering.labels_):
        if c == -1:
            continue
        clusters[c].append(aabbs[i])

    res_aabbs = []
    for curr_cluster in clusters.values():
        xmin = np.median([aabb.x_min for aabb in curr_cluster])
        xmax = np.median([aabb.x_max for aabb in curr_cluster])
        ymin = np.median([aabb.y_min for aabb in curr_cluster])
        ymax = np.median([aabb.y_max for aabb in curr_cluster])
        res_aabbs.append(BoundingBox(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax))
    return res_aabbs


def compute_dist_mat_2(
    aabbs1: List[BoundingBox], aabbs2: List[BoundingBox]
) -> np.ndarray:
    """Jaccard distance matrix of all pairs from ``aabbs1`` and ``aabbs2``."""
    dists = np.zeros((len(aabbs1), len(aabbs2)))
    for i in range(len(aabbs1)):
        for j in range(len(aabbs2)):
            dists[i, j] = 1 - compute_iou(aabbs1[i], aabbs2[j])
    return dists


def binary_classification_metrics(
    gt_aabbs: List[BoundingBox], pred_aabbs: List[BoundingBox]
) -> dict:
    iou_thres = 0.7
    ious = 1 - compute_dist_mat_2(gt_aabbs, pred_aabbs)
    match_counter = (ious > iou_thres).astype(int)
    gt_counter = np.sum(match_counter, axis=1)
    pred_counter = np.sum(match_counter, axis=0)
    return {
        "tp": int(np.count_nonzero(pred_counter == 1)),
        "fp": int(np.count_nonzero(pred_counter == 0)),
        "fn": int(np.count_nonzero(gt_counter == 0)),
    }


def normalize_image_transform(image, bounding_boxes):
    """Normalise a grayscale image to roughly ``[-0.5, 0.5]``."""
    return (image / 255.0) - 0.5, bounding_boxes


def draw_bboxes_on_image(
    img: np.ndarray,
    aabbs: List[BoundingBox],
    denormalise: bool = True,
) -> np.ndarray:
    """Draw bounding boxes on a (possibly normalised, grayscale) image."""
    img = img.copy()
    if denormalise:
        img = ((img + 0.5) * 255).astype(np.uint8)
    is_grayscale = len(img.shape) == 2
    if is_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for aabb in aabbs:
        aabb = aabb.enlarge_to_int_grid().as_type(int)
        cv2.rectangle(
            img,
            (int(aabb.x_min), int(aabb.y_min)),
            (int(aabb.x_max), int(aabb.y_max)),
            (0, 0, 255),  # Red
            2,
        )
    return img
