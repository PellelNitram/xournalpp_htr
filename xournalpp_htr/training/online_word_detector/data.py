"""Benchmark documents as stroke pages with word-group labels."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xournalpp_htr.documents import get_document
from xournalpp_htr.xio import load_benchmark

# Same classes the benchmark scores as words (see `benchmark._TEXT_CLASSES`).
TEXT_CLASSES = {"word", "digit", "mathematical_expression"}

#: Label of strokes that belong to no text group.
NON_TEXT = -1


@dataclass
class StrokePage:
    """The strokes of one layer of one page, in writing order.

    ``labels[i]`` is the id of the text group stroke ``i`` belongs to, or
    `NON_TEXT`. Group ids are only unique within a page.
    """

    document: str
    page_index: int
    layer_index: int
    strokes: list[tuple[np.ndarray, np.ndarray]]  # (x, y) per stroke
    labels: np.ndarray

    @property
    def groups(self) -> list[frozenset[int]]:
        """Ground truth text groups as sets of stroke indices."""
        return [
            frozenset(np.flatnonzero(self.labels == g).tolist())
            for g in sorted(set(self.labels.tolist()) - {NON_TEXT})
        ]


def load_pages(dataset_version: str | None = None) -> list[StrokePage]:
    """Return every annotated layer of the benchmark as a `StrokePage`."""
    pages = []
    for sample in load_benchmark(dataset_version):
        pages.extend(_pages_of_sample(sample.xopp_path, sample.gt_path))
    return pages


def _pages_of_sample(xopp_path: Path, gt_path: Path) -> list[StrokePage]:
    document = get_document(xopp_path)
    with open(gt_path) as f:
        annotations = json.load(f)["annotations"]

    pages = []
    for page_index, page in enumerate(document.pages):
        for layer_index, layer in enumerate(page.layers):
            if not layer.strokes:
                continue
            labels = np.full(len(layer.strokes), NON_TEXT)
            group_id = 0
            for ann in annotations:
                if (
                    ann["class"] in TEXT_CLASSES
                    and ann["page_index"] == page_index
                    and ann["layer_index"] == layer_index
                ):
                    labels[ann["stroke_indices"]] = group_id
                    group_id += 1
            pages.append(
                StrokePage(
                    document=xopp_path.name,
                    page_index=page_index,
                    layer_index=layer_index,
                    strokes=[(s.x, s.y) for s in layer.strokes],
                    labels=labels,
                )
            )
    return pages
