"""Generic axis-aligned bounding box geometry.

Base-deps-only (numpy). Shared by the WordDetector training, its post-processing
decoder, and inference. See ADR 006 section 3.
"""

from typing import NamedTuple, Optional

import numpy as np


class ImageDimensions(NamedTuple):
    height: int
    width: int


class BoundingBox:
    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        label: Optional[str] = None,
    ):
        """Axis-aligned bounding box.

        ``(x_min, y_min)`` is the top-left corner, ``(x_max, y_max)`` the
        bottom-right corner. ``label`` is an optional class/text label.
        """
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.label: Optional[str] = label

    def translate(self, dx: float, dy: float) -> "BoundingBox":
        """Translate the bounding box by ``(dx, dy)``."""
        return BoundingBox(
            self.x_min + dx,
            self.y_min + dy,
            self.x_max + dx,
            self.y_max + dy,
            self.label,
        )

    def scale(self, sx: float, sy: float) -> "BoundingBox":
        """Scale the bounding box by ``sx`` and ``sy``."""
        return BoundingBox(
            self.x_min * sx,
            self.y_min * sy,
            self.x_max * sx,
            self.y_max * sy,
            self.label,
        )

    def as_type(self, new_type) -> "BoundingBox":
        return BoundingBox(
            new_type(self.x_min),
            new_type(self.y_min),
            new_type(self.x_max),
            new_type(self.y_max),
            self.label,
        )

    def scale_around_center(self, sx, sy) -> "BoundingBox":
        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2
        return BoundingBox(
            x_min=center_x - sx * (center_x - self.x_min),
            y_min=center_y - sy * (center_y - self.y_min),
            x_max=center_x + sx * (self.x_max - center_x),
            y_max=center_y + sy * (self.y_max - center_y),
            label=self.label,
        )

    def clip(self, clip_aabb: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            x_min=min(max(self.x_min, clip_aabb.x_min), clip_aabb.x_max),
            y_min=min(max(self.y_min, clip_aabb.y_min), clip_aabb.y_max),
            x_max=max(min(self.x_max, clip_aabb.x_max), clip_aabb.x_min),
            y_max=max(min(self.y_max, clip_aabb.y_max), clip_aabb.y_min),
            label=self.label,
        )

    def area(self) -> float:
        """Return the area of the bounding box."""
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def enlarge_to_int_grid(self) -> "BoundingBox":
        return BoundingBox(
            x_min=np.floor(self.x_min),
            y_min=np.floor(self.y_min),
            x_max=np.ceil(self.x_max),
            y_max=np.ceil(self.y_max),
            label=self.label,
        )

    def enlarge(self, margin_x: float, margin_y: float) -> "BoundingBox":
        return BoundingBox(
            x_min=self.x_min - margin_x,
            x_max=self.x_max + margin_x,
            y_min=self.y_min - margin_y,
            y_max=self.y_max + margin_y,
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x_min={self.x_min}, y_min={self.y_min}, "
            f"x_max={self.x_max}, y_max={self.y_max}, label={self.label})"
        )
