"""Recognition must not depend on stroke color (issue #164).

The pipelines render each page to an image on a white canvas. Pale strokes,
as used on dark page themes, would vanish there, so every pipeline has to
render its pages with black strokes.
"""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from xournalpp_htr import models
from xournalpp_htr.documents import Document, StrokesDocument

PIPELINES = [
    "2024-07-18_htr_pipeline",
    "2026-06-07_htr_pipeline_native",
    "2026-09-02_yolo_detector",
    "2026-09-17_rf_detr_detector",
    "2026-10-05_yolo_detector_beam_vocab",
    "2026-10-05_yolo_detector_trocr",
    "2026-10-05_yolo_detector_trocr_large",
    "2026-10-09_yolo_detector_trocr_large_strip_punct",
    "2026-10-09_yolo_detector_ppocrv5",
    "2026-10-09_ppocrv5_det_rec",
    "2026-10-17_yolo_detector_simple_htr_ondb",
    "2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab",
    "2026-11-03_online_word_detector_hub_simple_htr_ondb",
]

MODEL_CLASSES = [
    "OnlineWordDetectorModel",
    "PPOCRv5Model",
    "RFDETRWordDetectorModel",
    "SimpleHTRModel",
    "SimpleHTROnDBModel",
    "TrOCRLargeModel",
    "TrOCRModel",
    "WordDetectorModel",
    "YOLOWordDetectorModel",
]


@pytest.mark.parametrize("pipeline_name", PIPELINES)
def test_pipeline_renders_pale_strokes_as_black(pipeline_name, monkeypatch):
    # No model weights needed: only the page render is of interest here.
    for name in MODEL_CLASSES:
        monkeypatch.setattr(models, name, MagicMock())
    monkeypatch.setattr(models, "read_page", lambda *args, **kwargs: [])

    renders = []
    original = Document.save_page_as_image

    def recording_render(self, page_index, out_path, *args, **kwargs):
        written = original(self, page_index, out_path, *args, **kwargs)
        renders.append(cv2.imread(str(written), cv2.IMREAD_GRAYSCALE))
        return written

    monkeypatch.setattr(Document, "save_page_as_image", recording_render)

    x = np.linspace(10, 90, 20)
    # Pale yellow, as on a dark page theme.
    document = StrokesDocument(
        [(x, np.full_like(x, 30.0))], width=100, height=60, color="#f1fa8c"
    )

    models.compute_predictions(pipeline_name, document)

    assert renders, "pipeline did not render a page"
    assert renders[0].min() < 64, "pale strokes must be rendered black"
