"""The stroke page type shared by training and inference. Numpy only."""

from dataclasses import dataclass

import numpy as np

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
