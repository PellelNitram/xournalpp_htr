"""The OnlineHTR pipeline reads each detected word straight from its strokes (issue #174)."""

from unittest.mock import MagicMock

import numpy as np

from xournalpp_htr import models
from xournalpp_htr.documents import Document, StrokesDocument

PIPELINE = "2026-11-22_online_word_detector_online_htr"


def _document():
    x = np.linspace(10, 40, 20)
    strokes = [(x, np.full_like(x, 30.0)), (x + 50, np.full_like(x, 32.0))]
    return StrokesDocument(strokes, width=120, height=60)


def test_pipeline_recognises_each_word_from_its_strokes(monkeypatch, tmp_path):
    detector = MagicMock()
    detector.group_strokes.return_value = [frozenset({0}), frozenset({1})]
    monkeypatch.setattr(models, "OnlineWordDetectorModel", MagicMock())
    models.OnlineWordDetectorModel.from_pretrained.return_value = detector

    recognizer = MagicMock()
    recognizer.recognize.side_effect = ["hello", "world"]
    monkeypatch.setattr(models, "OnlineHTRModel", MagicMock())
    models.OnlineHTRModel.from_local.return_value = recognizer

    def fail(*args, **kwargs):
        raise AssertionError("OnlineHTR needs no page render")

    monkeypatch.setattr(Document, "save_page_as_image", fail)
    monkeypatch.setenv(models.ONLINE_HTR_DIR_ENV, str(tmp_path))

    predictions = models.compute_predictions(PIPELINE, _document())

    models.OnlineHTRModel.from_local.assert_called_once_with(tmp_path)
    assert [p.text for p in predictions[0]] == ["hello", "world"]
    first, second = predictions[0]
    assert first.xmax < second.xmin  # the boxes follow the strokes of each word
    words = [call.args[0] for call in recognizer.recognize.call_args_list]
    assert [len(w) for w in words] == [1, 1]  # one stroke per word


def test_pipeline_loads_from_hub_without_a_local_dir(monkeypatch):
    detector = MagicMock()
    detector.group_strokes.return_value = []
    monkeypatch.setattr(models, "OnlineWordDetectorModel", MagicMock())
    models.OnlineWordDetectorModel.from_pretrained.return_value = detector
    monkeypatch.setattr(models, "OnlineHTRModel", MagicMock())
    monkeypatch.delenv(models.ONLINE_HTR_DIR_ENV, raising=False)

    models.compute_predictions(PIPELINE, _document())

    models.OnlineHTRModel.from_pretrained.assert_called_once_with()


def test_beam_vocab_pipeline_decodes_with_beam_search_and_the_word_list(
    monkeypatch, tmp_path
):
    detector = MagicMock()
    detector.group_strokes.return_value = [frozenset({0})]
    monkeypatch.setattr(models, "OnlineWordDetectorModel", MagicMock())
    models.OnlineWordDetectorModel.from_pretrained.return_value = detector
    monkeypatch.setattr(models, "OnlineHTRModel", MagicMock())
    recognizer = models.OnlineHTRModel.from_local.return_value
    recognizer.recognize.return_value = "cat"
    monkeypatch.setattr(models, "SimpleHTRModel", MagicMock())
    models.SimpleHTRModel.load_vocabulary.return_value = ["cat", "dog"]
    monkeypatch.setenv(models.ONLINE_HTR_DIR_ENV, str(tmp_path))

    models.compute_predictions(
        "2026-11-22_online_word_detector_online_htr_beam_vocab", _document()
    )

    recognizer.use_lexicon.assert_called_once_with(["cat", "dog"])
    assert recognizer.recognize.call_args.kwargs == {"decoder": "beam"}
