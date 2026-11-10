"""Tests for the HFHubInferenceModel ABC and WordDetectorModel (ADR 006)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from xournalpp_htr.inference_models import (
    HFHubInferenceModel,
    SimpleHTRModel,
    SimpleHTROnDBModel,
    WordDetectorModel,
    strip_appended_punctuation,
)
from xournalpp_htr.training.shared.bounding_box import BoundingBox

CHECKPOINT = (
    Path(__file__).parents[1]
    / "xournalpp_htr"
    / "training"
    / "word_detector"
    / "best_model.pth"
)

SIMPLE_HTR_CHECKPOINT = (
    Path(__file__).parents[1]
    / "xournalpp_htr"
    / "training"
    / "simple_htr"
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


def test_simple_htr_repo_id_follows_adr006_naming():
    assert SimpleHTRModel.HF_REPO_ID == "PellelNitram/xournalpp-htr-simple-htr"
    assert re.fullmatch(
        r"PellelNitram/xournalpp-htr-[a-z0-9-]+", SimpleHTRModel.HF_REPO_ID
    )


def test_simple_htr_ondb_repo_id_follows_adr006_naming():
    assert SimpleHTROnDBModel.HF_REPO_ID == "PellelNitram/xournalpp-htr-simple-htr-ondb"
    assert re.fullmatch(
        r"PellelNitram/xournalpp-htr-[a-z0-9-]+", SimpleHTROnDBModel.HF_REPO_ID
    )


def test_simple_htr_ondb_reuses_the_simple_htr_vocabulary(monkeypatch):
    # The word list is published next to SimpleHTR, not next to the OnDB model.
    monkeypatch.setattr(
        SimpleHTRModel,
        "load_vocabulary",
        classmethod(lambda cls, revision="main": ["word"]),
    )
    assert SimpleHTROnDBModel.load_vocabulary(revision="v1") == ["word"]


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


@pytest.mark.slow
@pytest.mark.skipif(
    not SIMPLE_HTR_CHECKPOINT.exists(),
    reason="best_model.pth is gitignored / local-only; skip ONNX round-trip in CI",
)
def test_simple_htr_onnx_roundtrip_offline(tmp_path: Path):
    import onnxruntime as ort

    pytest.importorskip(
        "tensorboard",
        reason="needs the training-simple-htr extra to export the checkpoint",
    )
    from xournalpp_htr.training.simple_htr.export import export

    paths = export(SIMPLE_HTR_CHECKPOINT, tmp_path)
    assert paths["onnx"].exists() and paths["config"].exists()

    with open(paths["config"]) as f:
        config = json.load(f)

    model = SimpleHTRModel(
        session=ort.InferenceSession(str(paths["onnx"])),
        config=config,
        revision="local",
    )
    assert "simple-htr" in repr(model)

    h, w = 32, 100
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(h, w), dtype=np.uint8)

    text = model.recognize(img)
    assert isinstance(text, str)


# --- TrOCR post-processing ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("words .", "words"),
        ("Thanks ,", "Thanks"),
        ("issues . ,", "issues"),
        ('say "', "say"),
        ("thoughts:", "thoughts:"),  # attached punctuation is kept
        ("Xournal++", "Xournal++"),
        ("li be", "li be"),  # only punctuation is stripped, not words
        ("15 .", "15"),
        (".", "."),
        ("", ""),
    ],
)
def test_strip_appended_punctuation(text, expected):
    assert strip_appended_punctuation(text) == expected


# --- PP-OCRv5 ---


@pytest.fixture(scope="module")
def ppocrv5_model():
    pytest.importorskip("rapidocr", reason="needs the `ppocr` extra")
    from xournalpp_htr.inference_models import PPOCRv5Model

    return PPOCRv5Model.from_pretrained()


@pytest.mark.slow
def test_ppocrv5_preprocess_keeps_aspect_ratio(ppocrv5_model):
    img = np.full((96, 200), 255, dtype=np.uint8)
    out = ppocrv5_model.preprocess(img)
    assert out.shape == (48, 100)


@pytest.mark.slow
def test_ppocrv5_recognize_returns_string(ppocrv5_model):
    img = np.full((48, 150), 255, dtype=np.uint8)
    assert isinstance(ppocrv5_model.recognize(img), str)


@pytest.mark.slow
def test_ppocrv5_read_page_blank_page_is_empty(ppocrv5_model):
    img = np.full((200, 400), 255, dtype=np.uint8)
    assert ppocrv5_model.read_page(img) == []


@pytest.mark.slow
def test_ppocrv5_read_page_finds_printed_words(ppocrv5_model):
    import cv2

    img = np.full((120, 600), 255, dtype=np.uint8)
    cv2.putText(img, "Hello big world", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    words = ppocrv5_model.read_page(img)
    assert "big" in [text for text, _ in words]
    assert all(isinstance(box, BoundingBox) for _, box in words)


# --- LFM3-VL ---


@pytest.fixture
def fake_llama_server():
    """A minimal stand-in for ``llama-server``: /health and chat completions."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply({"status": "ok"})

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            requests.append((self.path, json.loads(self.rfile.read(length))))
            self._reply({"choices": [{"message": {"content": " brain \n"}}]})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()


def test_lfm3_vl_recognize_sends_image_and_strips_reply(fake_llama_server):
    from xournalpp_htr.inference_models import LFM3VLModel

    url, requests = fake_llama_server
    text = LFM3VLModel(url).recognize(np.full((40, 80), 255, dtype=np.uint8))

    assert text == "brain"
    path, body = requests[0]
    assert path == "/v1/chat/completions"
    content = body["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["temperature"] == 0


def test_lfm3_vl_recognize_batch_returns_one_text_per_image(fake_llama_server):
    from xournalpp_htr.inference_models import LFM3VLModel

    url, requests = fake_llama_server
    images = [np.full((40, 80), 255, dtype=np.uint8)] * 3
    assert LFM3VLModel(url).recognize_batch(images) == ["brain"] * 3
    assert len(requests) == 3


def test_lfm3_vl_from_pretrained_reads_server_url_from_env(
    fake_llama_server, monkeypatch
):
    from xournalpp_htr.inference_models import LFM3VLModel

    url, _ = fake_llama_server
    monkeypatch.setenv(LFM3VLModel.SERVER_URL_ENV_VAR, url)
    assert LFM3VLModel.from_pretrained().server_url == url


def test_lfm3_vl_from_pretrained_explains_missing_server(monkeypatch):
    from xournalpp_htr.inference_models import LFM3VLModel

    monkeypatch.setenv(LFM3VLModel.SERVER_URL_ENV_VAR, "http://127.0.0.1:1")
    with pytest.raises(ConnectionError, match="llama-server"):
        LFM3VLModel.from_pretrained()


def test_lfm3_vl_read_page_maps_normalised_boxes_to_pixels(fake_llama_server):
    from xournalpp_htr.inference_models import LFM3VLModel

    url, requests = fake_llama_server
    model = LFM3VLModel(url)
    model._chat = (
        lambda *args, **kwargs: (
            '[{"t": "hi", "b": [100, 200, 300, 400]}, '
            '{"t": "", "b": [0, 0, 10, 10]}, '  # empty text: dropped
            '{"t": "flat", "b": [500, 500, 500, 600]}, '  # zero width: dropped
            '{"t": "edge", "b": [900, 900, 1200, 1000]}]'  # clamped to 1000
        )
    )
    page = np.full((200, 400), 255, dtype=np.uint8)
    page[90:110, 50:350] = 0  # some ink, else the page is skipped
    words = model.read_page(page)

    assert [text for text, _ in words] == ["hi", "edge"]
    box = words[0][1]
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (40.0, 40.0, 120.0, 80.0)
    assert words[1][1].x_max == 400.0


def test_lfm3_vl_read_page_downscales_large_pages_and_sends_grammar(
    fake_llama_server,
):
    import base64

    import cv2

    from xournalpp_htr.inference_models import LFM3VLModel

    url, requests = fake_llama_server
    model = LFM3VLModel(url)
    page = np.full((100, 4096), 255, dtype=np.uint8)
    page[40:60, 100:300] = 0  # some ink, else the strip is skipped
    assert model.read_page(page) == []  # the fake server replies "brain": no JSON

    _, body = requests[0]
    assert body["grammar"] == LFM3VLModel.PAGE_GRAMMAR
    data_url = body["messages"][0]["content"][0]["image_url"]["url"]
    png = base64.b64decode(data_url.split(",", 1)[1])
    sent = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert max(sent.shape) == LFM3VLModel.PAGE_MAX_IMAGE_DIM


def test_lfm3_vl_parse_page_reply_keeps_complete_entries_of_cut_off_reply():
    from xournalpp_htr.inference_models import LFM3VLModel

    cut = '[{"t": "a", "b": [1, 2, 3, 4]}, {"t": "b", "b": [5, 6'
    assert LFM3VLModel._parse_page_reply(cut) == [{"t": "a", "b": [1, 2, 3, 4]}]
    assert LFM3VLModel._parse_page_reply("not json") == []


def test_lfm3_vl_read_page_reads_in_strips_and_finds_overlap_words_once():
    from xournalpp_htr.inference_models import LFM3VLModel

    model = LFM3VLModel("http://unused")
    strip_height = model.PAGE_STRIP_HEIGHT
    stride = strip_height - model.PAGE_STRIP_OVERLAP
    # Two strips: 0..strip_height and stride..stride + strip_height.
    page = np.full((stride + strip_height, 400), 255, dtype=np.uint8)
    page[::50] = 0  # ink everywhere, so no strip is skipped
    calls = []

    # One word in the overlap of the two strips.
    word_top = stride + 10
    word_bottom = word_top + 40

    def fake_read_strip(strip):
        calls.append(strip.shape[0])
        top = 0 if len(calls) == 1 else stride
        return [("mid", BoundingBox(0, word_top - top, 100, word_bottom - top))]

    model._read_strip = fake_read_strip
    words = model.read_page(page)

    assert calls == [strip_height, strip_height]
    assert [text for text, _ in words] == ["mid"]  # not found twice
    box = words[0][1]
    assert (box.y_min, box.y_max) == (word_top, word_bottom)


def test_lfm3_vl_read_page_skips_strips_without_ink():
    from xournalpp_htr.inference_models import LFM3VLModel

    model = LFM3VLModel("http://unused")
    model._read_strip = lambda strip: pytest.fail("blank strip was sent")
    assert model.read_page(np.full((1000, 400), 255, dtype=np.uint8)) == []


def test_lfm3_vl_read_strip_drops_duplicate_boxes():
    from xournalpp_htr.inference_models import LFM3VLModel

    model = LFM3VLModel("http://unused")
    model._chat = (
        lambda *args, **kwargs: (
            '[{"t": "a", "b": [100, 100, 300, 300]}, '
            '{"t": "a", "b": [110, 105, 310, 305]}, '  # same box again: dropped
            '{"t": "b", "b": [600, 100, 800, 300]}]'
        )
    )
    words = model._read_strip(np.full((100, 400), 255, dtype=np.uint8))
    assert [text for text, _ in words] == ["a", "b"]
