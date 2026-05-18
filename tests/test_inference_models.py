"""Tests for the HFHubInferenceModel ABC and WordDetectorModel (ADR 006)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from xournalpp_htr.inference_models import HFHubInferenceModel, WordDetectorModel
from xournalpp_htr.training.shared.bounding_box import BoundingBox

CHECKPOINT = (
    Path(__file__).parents[1]
    / "xournalpp_htr"
    / "training"
    / "word_detector"
    / "best_model.pth"
)


# --- ABC contract ---


def test_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        HFHubInferenceModel(revision="main")


def test_subclass_without_from_pretrained_is_abstract():
    class Incomplete(HFHubInferenceModel):
        HF_REPO_ID = "x/y"

    with pytest.raises(TypeError):
        Incomplete(revision="main")


def test_concrete_subclass_stores_revision_and_repr():
    class Dummy(HFHubInferenceModel):
        HF_REPO_ID = "PellelNitram/xournalpp-htr-dummy"

        @classmethod
        def from_pretrained(cls, revision: str = "main"):
            return cls(revision)

    m = Dummy.from_pretrained(revision="v1.2.0")
    assert m.revision == "v1.2.0"
    assert repr(m) == (
        "Dummy(repo='PellelNitram/xournalpp-htr-dummy', revision='v1.2.0')"
    )


def test_word_detector_repo_id_follows_adr006_naming():
    # ADR 006 section 4: PellelNitram/xournalpp-htr-<model-name>
    assert WordDetectorModel.HF_REPO_ID == "PellelNitram/xournalpp-htr-word-detector"
    assert re.fullmatch(
        r"PellelNitram/xournalpp-htr-[a-z0-9-]+", WordDetectorModel.HF_REPO_ID
    )


# --- ONNX round-trip (offline, uses the local checkpoint) ---


@pytest.mark.slow
@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="best_model.pth is gitignored / local-only; skip ONNX round-trip in CI",
)
def test_word_detector_onnx_roundtrip_offline(tmp_path: Path):
    import onnxruntime as ort

    # Export lives in the guarded training subpackage (ADR 006 section 3);
    # skip when the training extra is not installed (e.g. lean inference env).
    pytest.importorskip(
        "tensorboard",
        reason="needs the training-word-detector extra to export the checkpoint",
    )
    from xournalpp_htr.training.word_detector.export import export

    paths = export(CHECKPOINT, tmp_path)
    assert paths["onnx"].exists() and paths["config"].exists()

    with open(paths["config"]) as f:
        config = json.load(f)

    model = WordDetectorModel(
        session=ort.InferenceSession(str(paths["onnx"])),
        config=config,
        revision="local",
    )
    assert "word-detector" in repr(model)

    # Detection on an arbitrary-size synthetic image must not crash and must
    # return boxes within the *passed* image's pixel bounds.
    h, w = 200, 320
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(h, w), dtype=np.uint8)

    boxes = model.detect(img)
    assert isinstance(boxes, list)
    for b in boxes:
        assert isinstance(b, BoundingBox)
        assert np.isfinite([b.x_min, b.y_min, b.x_max, b.y_max]).all()
        assert -1 <= b.x_min and b.x_max <= w + 1
        assert -1 <= b.y_min and b.y_max <= h + 1
