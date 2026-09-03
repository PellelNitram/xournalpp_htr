"""YOLO-based word detector for use in the benchmark pipeline.

Wraps a local YOLO .pt checkpoint and exposes a ``detect()`` method
compatible with the interface used by ``compute_predictions`` in
``xournalpp_htr.models``.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from xournalpp_htr.training.shared.bounding_box import BoundingBox


class YOLOWordDetector:
    def __init__(self, weights: Path, conf: float = 0.25, imgsz: int = 1024):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_grayscale: np.ndarray) -> List[BoundingBox]:
        if len(image_grayscale.shape) == 2:
            img = cv2.cvtColor(image_grayscale, cv2.COLOR_GRAY2BGR)
        else:
            img = image_grayscale

        results = self.model.predict(
            img, conf=self.conf, imgsz=self.imgsz, verbose=False
        )

        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append(BoundingBox(float(x1), float(y1), float(x2), float(y2)))
        return boxes
