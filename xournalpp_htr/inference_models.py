"""HuggingFace Hub inference model loading (ADR 006).

Custom models are stored on HF Hub as plain artifacts (ONNX export plus
supporting files) rather than via ``transformers``/``PreTrainedModel``. Every
inference model implements :class:`HFHubInferenceModel`, giving consumers a
uniform, parameter-free ``from_pretrained()`` loading interface without
depending on ``transformers``.

The ABC deliberately does **not** define ``predict()``/``__call__()``: the
inference signature varies too much across models. The central inference API is
``compute_predictions(document, pipeline)`` (ADR 003); the ABC's responsibility
is model lifecycle (loading and version introspection) only.
"""

import base64
import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from xournalpp_htr.training.shared.bounding_box import BoundingBox
from xournalpp_htr.training.shared.ctc_decoding import beam_decode, build_beam_decoder
from xournalpp_htr.training.shared.postprocessing import (
    cluster_aabbs,
    decode,
    fg_by_cc,
    normalize_image_transform,
)
from xournalpp_htr.training.word_detector_online.features import (
    consecutive_pair_features,
)
from xournalpp_htr.training.word_detector_online.model import (
    TreeEnsemble,
    groups_from_merges,
)
from xournalpp_htr.training.word_detector_online.strokes import NON_TEXT, StrokePage


class HFHubInferenceModel(ABC):
    """Base class binding an inference model to its HF Hub repository.

    ``HF_REPO_ID`` binds each subclass to its repository. ``revision`` is stored
    on the instance so callers can introspect which version is loaded (useful
    for logging and reproducibility); the default ``__repr__`` surfaces both.
    """

    HF_REPO_ID: ClassVar[str]

    def __init__(self, revision: str):
        self.revision = revision

    @classmethod
    @abstractmethod
    def from_pretrained(cls, revision: str = "main") -> "HFHubInferenceModel": ...

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(repo={self.HF_REPO_ID!r}, "
            f"revision={self.revision!r})"
        )


class WordDetectorModel(HFHubInferenceModel):
    """WordDetectorNN word-detection model, loaded from HF Hub as ONNX.

    The repository contains ``model.onnx`` (the network with softmax baked in)
    and ``config.json`` (pre/post-processing parameters). Inference runs the
    ONNX graph with ``onnxruntime`` and decodes the segmentation/geometry maps
    into clustered word bounding boxes. This model performs word *detection*
    only -- it produces no transcription.
    """

    HF_REPO_ID = "PellelNitram/xournalpp-htr-word-detector"

    DEFAULT_SCALE = 0.4

    def __init__(self, session: ort.InferenceSession, config: dict, revision: str):
        super().__init__(revision)
        self.session = session
        self.config = config
        self._input_name = session.get_inputs()[0].name

    @classmethod
    def from_pretrained(cls, revision: str = "main") -> "WordDetectorModel":
        onnx_path = hf_hub_download(cls.HF_REPO_ID, "model.onnx", revision=revision)
        config_path = hf_hub_download(cls.HF_REPO_ID, "config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)
        return cls(
            session=ort.InferenceSession(onnx_path),
            config=config,
            revision=revision,
        )

    @staticmethod
    def _ceil32(val: int) -> int:
        return val if val % 32 == 0 else (val // 32 + 1) * 32

    def detect(
        self, image_grayscale: np.ndarray, scale: float | None = None
    ) -> List[BoundingBox]:
        """Detect word bounding boxes in a grayscale image.

        The image is scaled by *scale* (default 0.4) and padded to multiples of
        32, preserving aspect ratio. The returned boxes are in the pixel
        coordinate system of the *passed* image.
        """
        if scale is None:
            scale = self.DEFAULT_SCALE

        orig_h, orig_w = image_grayscale.shape[:2]

        # Pre-processing: scale, pad to multiples of 32, normalise.
        img_scaled = cv2.resize(image_grayscale, None, fx=scale, fy=scale)
        padded_h = self._ceil32(img_scaled.shape[0])
        padded_w = self._ceil32(img_scaled.shape[1])
        img_padded = np.ones((padded_h, padded_w), dtype=np.uint8) * 255
        img_padded[: img_scaled.shape[0], : img_scaled.shape[1]] = img_scaled
        normalised, _ = normalize_image_transform(img_padded, None)
        net_input = normalised.astype(np.float32)[None, None, :, :]

        # Inference (softmax is baked into the exported ONNX graph).
        output = self.session.run(None, {self._input_name: net_input})[0]
        output = output[0]  # drop batch dim -> (NUM_MAPS, out_h, out_w)

        # Post-processing: decode maps -> scale back -> clip -> cluster.
        decoded = decode(
            output,
            scale=net_input.shape[2] / output.shape[1],
            comp_fg=fg_by_cc(
                thres=self.config["fg_cc_threshold"],
                max_num=self.config["fg_cc_max_num"],
            ),
        )
        decoded = [aabb.scale(1 / scale, 1 / scale) for aabb in decoded]
        clip_box = BoundingBox(0, 0, orig_w - 1, orig_h - 1)
        clustered = cluster_aabbs([aabb.clip(clip_box) for aabb in decoded])

        return clustered


class SimpleHTRModel(HFHubInferenceModel):
    """SimpleHTR word-recognition model, loaded from HF Hub as ONNX.

    The repository contains ``model.onnx`` (the CNN+LSTM+CTC network) and
    ``config.json`` (charset, input dimensions, normalisation). Inference runs
    the ONNX graph with ``onnxruntime`` and decodes the CTC output into text.
    """

    HF_REPO_ID = "PellelNitram/xournalpp-htr-simple-htr"

    def __init__(self, session: ort.InferenceSession, config: dict, revision: str):
        super().__init__(revision)
        self.session = session
        self.config = config
        self._input_name = session.get_inputs()[0].name
        self._charset = config["charset"]
        self._beam_decoder = build_beam_decoder(self._charset)
        self._vocabulary: set[str] | None = None

    @classmethod
    def from_pretrained(cls, revision: str = "main") -> "SimpleHTRModel":
        onnx_path = hf_hub_download(cls.HF_REPO_ID, "model.onnx", revision=revision)
        config_path = hf_hub_download(cls.HF_REPO_ID, "config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)
        return cls(
            session=ort.InferenceSession(onnx_path),
            config=config,
            revision=revision,
        )

    @classmethod
    def from_local(cls, model_dir: Path) -> "SimpleHTRModel":
        """Load an exported model from a local directory.

        For benchmarking a retrained model before publishing it. ``model_dir``
        holds ``model.onnx`` and ``config.json``, as written by the model's
        ``export.py``. The loaded model reports ``revision="local"``.
        """
        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        return cls(
            session=ort.InferenceSession(str(model_dir / "model.onnx")),
            config=config,
            revision="local",
        )

    @classmethod
    def load_vocabulary(cls, revision: str = "main") -> list[str]:
        """Download the word list published next to the model on HF Hub.

        Built and uploaded by ``build_vocabulary.py``; pass it to
        `use_lexicon` for lexicon-biased "beam" decoding.
        """
        path = hf_hub_download(cls.HF_REPO_ID, "vocabulary.txt", revision=revision)
        return Path(path).read_text().split()

    def use_lexicon(self, vocabulary: list[str] | None) -> None:
        """Bias "beam" decoding towards a word list (see build_vocabulary.py).

        Pass ``None`` to go back to plain beam search. Has no effect on
        greedy decoding.
        """
        self._vocabulary = (
            {w.lower() for w in vocabulary} if vocabulary is not None else None
        )

    def preprocess(self, image_grayscale: np.ndarray) -> np.ndarray:
        """Resize a grayscale word image to the network's input dimensions
        (uniform scale, centered on white canvas), before normalisation.

        Returns:
            (height, width) uint8 image, exactly what the network sees.
        """
        input_size = self.config["input_size"]
        in_h, in_w = input_size["height"], input_size["width"]

        h, w = image_grayscale.shape[:2]
        scale = min(in_w / w, in_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image_grayscale, (new_w, new_h))

        canvas = np.ones((in_h, in_w), dtype=np.uint8) * 255
        y_off = (in_h - new_h) // 2
        x_off = (in_w - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return canvas

    def _compute_log_probs(self, image_grayscale: np.ndarray) -> np.ndarray:
        """Preprocess a grayscale word image and run the ONNX network.

        Returns:
            (seq_len, num_classes) log-probabilities.
        """
        norm = self.config["normalization"]
        canvas = self.preprocess(image_grayscale)

        normalised = canvas.astype(np.float32) / norm["scale"] + norm["shift"]
        net_input = normalised[None, None, :, :]

        log_probs = self.session.run(None, {self._input_name: net_input})[0]
        return log_probs[
            :, 0, :
        ]  # (seq_len, batch, num_classes) -> (seq_len, num_classes)

    def recognize(self, image_grayscale: np.ndarray, decoder: str = "greedy") -> str:
        """Recognise text in a grayscale word image.

        Args:
            image_grayscale: grayscale word image.
            decoder: "greedy" (default) or "beam" CTC decoding.
        """
        if decoder not in ("greedy", "beam"):
            raise ValueError(
                f"Unknown decoder {decoder!r}, expected 'greedy' or 'beam'."
            )

        log_probs = self._compute_log_probs(image_grayscale)

        if decoder == "beam":
            return beam_decode(
                log_probs, self._beam_decoder, vocabulary=self._vocabulary
            )

        predictions = log_probs.argmax(axis=1)

        blank = len(self._charset)
        chars = []
        prev = blank
        for idx in predictions:
            if idx != prev and idx != blank:
                chars.append(self._charset[idx])
            prev = idx
        return "".join(chars)


class SimpleHTROnDBModel(SimpleHTRModel):
    """SimpleHTR retrained on IAM-DB words plus rendered IAM-OnDB crops.

    Same network, preprocessing and decoding as `SimpleHTRModel`; only the
    weights differ (issue #150, see ``docs/models/simple_htr_ondb.md``). The
    word list for lexicon-biased decoding is not duplicated here, it is the one
    published next to the original model.
    """

    HF_REPO_ID = "PellelNitram/xournalpp-htr-simple-htr-ondb"

    @classmethod
    def load_vocabulary(cls, revision: str = "main") -> list[str]:
        """Word list of `SimpleHTRModel`; ``revision`` is ignored."""
        return SimpleHTRModel.load_vocabulary()


# Punctuation TrOCR appends after a space, e.g. "words ." or "Thanks ,".
_APPENDED_PUNCTUATION = re.compile(r"""(\s+[.,;:!?'"]+)+$""")


def strip_appended_punctuation(text: str) -> str:
    """Remove punctuation that TrOCR appends to a word after a space.

    The IAM checkpoints were finetuned on text lines, where punctuation is
    space-separated, so on single word crops they often add a sentence-final
    " ." or " ," (issue #156). Punctuation attached to the word, as in
    "thoughts:", is kept.
    """
    return _APPENDED_PUNCTUATION.sub("", text).strip()


class TrOCRModel(HFHubInferenceModel):
    """Pretrained TrOCR word-recognition model, run with ``transformers``.

    Exception to ADR 006: this is Microsoft's IAM-finetuned checkpoint, not
    our own ONNX export, so we can measure whether TrOCR beats SimpleHTR
    before investing in an export (issue #156). Needs the ``trocr`` extra;
    ``transformers`` and ``torch`` are imported lazily so the base install
    stays lean.
    """

    HF_REPO_ID = "microsoft/trocr-base-handwritten"
    # Pinned so the pipelines keep loading the benchmarked files: we don't
    # control Microsoft's repo, so `main` could change underneath us.
    REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"  # main on 2026-10-05

    # Generous for a single word crop, but leaves room for crops where the
    # detector merged several words.
    MAX_NEW_TOKENS = 32

    def __init__(self, processor, model, device: str, revision: str):
        super().__init__(revision)
        self.processor = processor
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, revision: str | None = None) -> "TrOCRModel":
        """Load the checkpoint at ``revision``, by default the pinned
        `REVISION`."""
        revision = revision or cls.REVISION
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                RobertaTokenizer,
                TrOCRProcessor,
                VisionEncoderDecoderModel,
            )
        except ImportError as e:
            raise ImportError(
                "TrOCRModel needs the `trocr` extra: `uv sync --extra trocr`."
            ) from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Built from its parts: transformers 5 can't auto-load the tokenizer
        # of this legacy repo (vocab.json + merges.txt, no tokenizer.json).
        processor = TrOCRProcessor(
            image_processor=AutoImageProcessor.from_pretrained(
                cls.HF_REPO_ID, revision=revision
            ),
            tokenizer=RobertaTokenizer.from_pretrained(
                cls.HF_REPO_ID, revision=revision
            ),
        )
        model = VisionEncoderDecoderModel.from_pretrained(
            cls.HF_REPO_ID, revision=revision
        )
        model.to(device).eval()
        return cls(processor=processor, model=model, device=device, revision=revision)

    def preprocess(self, image_grayscale: np.ndarray) -> np.ndarray:
        """Resize a grayscale word image to the encoder's input size, as the
        processor does (no aspect ratio preservation), before normalisation.

        Returns:
            (height, width) uint8 image, what the network sees (in grayscale).
        """
        size = self.processor.image_processor.size
        return cv2.resize(image_grayscale, (size["width"], size["height"]))

    def recognize_batch(
        self, images_grayscale: list[np.ndarray], batch_size: int = 16
    ) -> list[str]:
        """Recognise text in grayscale word images, ``batch_size`` at a time."""
        import torch

        texts = []
        for start in range(0, len(images_grayscale), batch_size):
            batch = [
                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                for image in images_grayscale[start : start + batch_size]
            ]
            pixel_values = self.processor(
                images=batch, return_tensors="pt"
            ).pixel_values.to(self.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    pixel_values, max_new_tokens=self.MAX_NEW_TOKENS
                )
            decoded = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            texts.extend(text.strip() for text in decoded)
        return texts

    def recognize(self, image_grayscale: np.ndarray) -> str:
        """Recognise text in a grayscale word image."""
        return self.recognize_batch([image_grayscale])[0]


class TrOCRLargeModel(TrOCRModel):
    """Largest handwritten TrOCR checkpoint (558M parameters vs. 334M for
    base); same tokenizer and inference as `TrOCRModel`."""

    HF_REPO_ID = "microsoft/trocr-large-handwritten"
    REVISION = "e68501f437cd2587ae5d68ee457964cac824ddee"  # main on 2026-10-05


class PPOCRv5Model:
    """PaddlePaddle's PP-OCRv5 text recognition, run as ONNX through ``rapidocr``.

    Pretrained and used as is (issue #157); like TrOCR (issue #156) it is not
    our own export, so it does not follow ADR 006's HF Hub layout. Weights are
    downloaded by ``rapidocr`` on first use. Needs the ``ppocr`` extra;
    ``rapidocr`` is imported lazily so the base install stays lean.

    Two modes, as the recogniser can run alone or behind PP-OCRv5's own text
    detector:

    - `recognize` / `recognize_batch`: recognition only, on word crops cut by
      another detector (e.g. YOLO).
    - `read_page`: detection and recognition on a whole page.
    """

    # Height the recognition network resizes its input to.
    REC_INPUT_HEIGHT = 48

    def __init__(self, rec_engine, page_engine):
        # Separate engines: a `RapidOCR` call with `use_det=False` leaves the
        # engine in recognition-only mode for later calls.
        self.rec_engine = rec_engine
        self.page_engine = page_engine

    @classmethod
    def from_pretrained(cls) -> "PPOCRv5Model":
        try:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        except ImportError as e:
            raise ImportError(
                "PPOCRv5Model needs the `ppocr` extra: `uv sync --extra ppocr`."
            ) from e

        params = {
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.lang_type": LangRec.EN,
            "Rec.model_type": ModelType.MOBILE,
        }
        return cls(RapidOCR(params=params), RapidOCR(params=params))

    def preprocess(self, image_grayscale: np.ndarray) -> np.ndarray:
        """Resize a grayscale word image to the recognition input height,
        keeping the aspect ratio.

        Returns:
            (REC_INPUT_HEIGHT, width) uint8 image, what the network sees (in
            grayscale).
        """
        h, w = image_grayscale.shape[:2]
        new_w = max(1, round(w * self.REC_INPUT_HEIGHT / h))
        return cv2.resize(image_grayscale, (new_w, self.REC_INPUT_HEIGHT))

    def recognize(self, image_grayscale: np.ndarray) -> str:
        """Recognise text in a grayscale word image, without detection."""
        result = self.rec_engine(
            image_grayscale, use_det=False, use_cls=False, use_rec=True
        )
        return " ".join(result.txts) if result.txts else ""

    def recognize_batch(self, images_grayscale: list[np.ndarray]) -> list[str]:
        return [self.recognize(image) for image in images_grayscale]

    def read_page(self, image_grayscale: np.ndarray) -> list[tuple[str, BoundingBox]]:
        """Detect and recognise words on a whole page.

        PP-OCRv5 detects text lines; ``rapidocr`` splits them into words, which
        are returned with axis-aligned boxes.
        """
        result = self.page_engine(image_grayscale, return_word_box=True)
        words = []
        if not result.txts:  # nothing detected; word_results is then degenerate
            return words
        for line in result.word_results or ():
            for text, _score, quad in line:
                if not text.strip() or quad is None:
                    continue
                xs = [point[0] for point in quad]
                ys = [point[1] for point in quad]
                words.append(
                    (
                        text,
                        BoundingBox(
                            x_min=float(min(xs)),
                            y_min=float(min(ys)),
                            x_max=float(max(xs)),
                            y_max=float(max(ys)),
                        ),
                    )
                )
        return words


class IndraModel:
    """Datalab's Indra OCR, run as ONNX through the ``indra-ocr`` package.

    Pretrained and used as is (issue #166); like PP-OCRv5 it is not our own
    export, so it does not follow ADR 006's HF Hub layout. Weights are
    downloaded by ``indra-ocr`` on first use. Needs the ``indra`` extra;
    ``indra_ocr`` is imported lazily so the base install stays lean.

    Indra reads whole pages (its own line detector and recogniser) and is
    built for printed documents. On word crops cut by another detector it
    does far worse, as it relies on the context of a line, so there is no
    word-recognition mode.
    """

    # Words are cut where the recogniser emits a space. Its tokens carry the x
    # position of their CTC peak, so a word spans from its first to its last
    # token, and the box is stretched to the midpoint of the gap to each
    # neighbouring word (the line's ends for the first and last word).

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def from_pretrained(cls) -> "IndraModel":
        try:
            import indra_ocr
        except ImportError as e:
            raise ImportError(
                "IndraModel needs the `indra` extra: `uv sync --extra indra`."
            ) from e

        # `Converter.ocr` is the ONNX engine; the converter itself reads PDFs.
        return cls(indra_ocr.load(threads=4).ocr)

    def read_page(self, image_grayscale: np.ndarray) -> list[tuple[str, BoundingBox]]:
        """Detect and recognise words on a whole page.

        Indra detects text lines; each is split into words at the recogniser's
        space tokens. Words come with axis-aligned boxes.
        """
        engine = self.engine
        boxes = engine.detect(image_grayscale)
        if len(boxes) == 0:
            return []

        words = []
        for line_box, (_text, _conf, tokens) in zip(
            boxes, engine.recognize(image_grayscale, boxes), strict=True
        ):
            if not tokens:  # line below the recogniser's confidence floor
                continue
            x_min, y_min, x_max, y_max = (float(v) for v in line_box[:4])

            segments: list[list] = [[]]
            for token in tokens:
                if engine._is_space(token[0]):
                    if segments[-1]:
                        segments.append([])
                else:
                    segments[-1].append(token)
            segments = [segment for segment in segments if segment]
            spans = [
                (min(t[1] for t in segment), max(t[1] for t in segment))
                for segment in segments
            ]

            for i, (segment, (start, end)) in enumerate(
                zip(segments, spans, strict=True)
            ):
                text = engine._decode(segment).strip()
                if not text:
                    continue
                left = (spans[i - 1][1] + start) / 2 if i > 0 else x_min
                right = (end + spans[i + 1][0]) / 2 if i + 1 < len(spans) else x_max
                words.append(
                    (
                        text,
                        BoundingBox(
                            x_min=max(x_min, left),
                            y_min=y_min,
                            x_max=min(x_max, right),
                            y_max=y_max,
                        ),
                    )
                )
        return words


class LFM3VLModel:
    """Liquid AI's LFM3-VL-2B vision-language model, run by a ``llama-server``.

    Pretrained and used as is (issue #171); not our own export, so it does not
    follow ADR 006's HF Hub layout. The model runs in a separate ``llama.cpp``
    server that we talk to over its OpenAI-compatible HTTP API, so this class
    needs no extra dependency. Start the server with::

        llama-server -hf LiquidAI/LFM3-VL-2B-GGUF:Q4_0 --reasoning off

    ``--reasoning off`` is required: otherwise the model spends its token
    budget thinking and returns an empty reply. ``-np 1 -c 16384 -cram 0``
    keeps the memory use low (the default 8 GiB prompt cache got a server
    OOM-killed), and ``-fa on -ctk q8_0 -ctv q8_0`` is a little faster. The server address is read
    from the ``XOURNALPP_HTR_LLAMA_SERVER_URL`` environment variable
    (default `DEFAULT_SERVER_URL`).
    """

    DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
    SERVER_URL_ENV_VAR = "XOURNALPP_HTR_LLAMA_SERVER_URL"

    PROMPT = "Transcribe the handwritten word in this image. Reply with the word only."

    # --- Replica of the Rnote fork's page reading (issue #171) -------------------
    # MichaelEssich/rnote, branch `feature/ocr`, commit 9f55d2c
    # (crates/rnote-engine/src/ocr.rs; GPL-3.0). One call per whole page with
    # that fork's prompt, grammar and sampling. The model reports boxes as
    # coordinates normalised to 0..1000 (image width/height).
    RNOTE_PROMPT = (
        "Transcribe every word of handwriting in this image, in reading order. "
        "Reply with ONLY a JSON array of objects, each with field t (the "
        "recognized word as a string) and field b (a 4-element array of numbers "
        "x0, y0, x1, y1 giving that word's bounding box as normalized "
        "coordinates in the range 0 to 1000, where x0 is 1000 times the left "
        "edge divided by the image width, y0 is 1000 times the top edge divided "
        "by the image height, x1 the right edge, y1 the bottom edge). "
        "Do not write any other text."
    )
    # GBNF grammar forcing exactly the JSON array the prompt describes. Hand-written
    # rather than `json_schema`: llama.cpp's generated grammars can abort sampler
    # setup (ggml-org/llama.cpp#22396).
    RNOTE_GRAMMAR = r"""root ::= "[" space (element ("," space element)*)? "]" space
element ::= "{" space "\"t\"" space ":" space string "," space "\"b\"" space ":" space "[" space number ("," space number){3} "]" space "}" space
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
space ::= [ \t\n]*
"""
    RNOTE_SAMPLING = {
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": 50,
        "min_p": 0.0,
        "repeat_penalty": 1.1,
    }
    RNOTE_MAX_NEW_TOKENS = 8192
    # A page takes minutes on CPU.
    RNOTE_TIMEOUT_SECONDS = 20 * 60
    # Longest page edge, in pixels, that the fork rasterizes pages to.
    RNOTE_MAX_IMAGE_DIM = 2048
    RNOTE_MIN_SCALE, RNOTE_MAX_SCALE = 0.5, 4.0

    # Generous for a single word crop, but leaves room for crops where the
    # detector merged several words.
    MAX_NEW_TOKENS = 32
    TIMEOUT_SECONDS = 300

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    @classmethod
    def from_pretrained(cls) -> "LFM3VLModel":
        """Connect to the running ``llama-server``; fail early if unreachable."""
        server_url = os.environ.get(cls.SERVER_URL_ENV_VAR, cls.DEFAULT_SERVER_URL)
        model = cls(server_url)
        try:
            with urllib.request.urlopen(
                f"{model.server_url}/health", timeout=10
            ) as response:
                response.read()
        except OSError as e:
            raise ConnectionError(
                f"LFM3VLModel could not reach a llama-server at {model.server_url}. "
                "Start one with `llama-server -hf LiquidAI/LFM3-VL-2B-GGUF:Q4_0 "
                f"--reasoning off`, or set {cls.SERVER_URL_ENV_VAR}."
            ) from e
        return model

    def preprocess(self, image_grayscale: np.ndarray) -> np.ndarray:
        """What the model sees: the crop itself, the server resizes it."""
        return image_grayscale

    def _chat(
        self, image_grayscale: np.ndarray, prompt: str, timeout: float, **params
    ) -> str:
        """Send one image with ``prompt`` to the server, return the reply text."""
        ok, png = cv2.imencode(".png", image_grayscale)
        if not ok:
            raise ValueError("Could not encode the image as PNG.")
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
            **params,
        }
        request = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.load(response)
        return reply["choices"][0]["message"]["content"].strip()

    def recognize(self, image_grayscale: np.ndarray) -> str:
        """Recognise the text in a grayscale word image."""
        return self._chat(
            image_grayscale,
            self.PROMPT,
            self.TIMEOUT_SECONDS,
            max_tokens=self.MAX_NEW_TOKENS,
        )

    def recognize_batch(self, images_grayscale: list[np.ndarray]) -> list[str]:
        return [self.recognize(image) for image in images_grayscale]

    @classmethod
    def rnote_scale(cls, page_width: float, page_height: float) -> float:
        """Raster scale (pixels per document unit) at which the fork renders a
        page: its longest edge about `RNOTE_MAX_IMAGE_DIM` pixels, clamped."""
        scale = cls.RNOTE_MAX_IMAGE_DIM / max(page_width, page_height, 1.0)
        return min(max(scale, cls.RNOTE_MIN_SCALE), cls.RNOTE_MAX_SCALE)

    def read_page_like_rnote(
        self, image_grayscale: np.ndarray
    ) -> list[tuple[str, BoundingBox]]:
        """Read a whole page in one model call, as the Rnote fork does.

        No strips, no duplicate removal and no snapping: only what the fork does
        with the reply. A reply that is not a valid JSON array (e.g. one cut off
        by the context limit) gives no words for the page, as the fork fails
        the page then.

        Returns:
            (text, box) pairs, with boxes in pixels of ``image_grayscale``.
        """
        height, width = image_grayscale.shape[:2]
        reply = self._chat(
            image_grayscale,
            self.RNOTE_PROMPT,
            self.RNOTE_TIMEOUT_SECONDS,
            max_tokens=self.RNOTE_MAX_NEW_TOKENS,
            grammar=self.RNOTE_GRAMMAR,
            **self.RNOTE_SAMPLING,
        )
        try:
            words_json = []
            for item in json.loads(reply):
                box = [float(v) for v in item["b"]]
                if len(box) != 4:
                    raise ValueError("box without 4 numbers")
                words_json.append((str(item["t"]), box))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"Rnote replica: unusable reply ({e}); no words for this page.")
            return []

        words = []
        for text, (x0, y0, x1, y1) in words_json:
            text = text.strip()
            if not text:
                continue
            x0, x1 = x0 / 1000 * width, x1 / 1000 * width
            y0, y1 = y0 / 1000 * height, y1 / 1000 * height
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            x0, x1 = min(max(x0, 0.0), width), min(max(x1, 0.0), width)
            y0, y1 = min(max(y0, 0.0), height), min(max(y1, 0.0), height)
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            words.append((text, BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)))
        return words


class YOLOWordDetectorModel(HFHubInferenceModel):
    """YOLO-based word-detection model, loaded from HF Hub as ONNX.

    The repository contains ``model.onnx`` (exported via ultralytics with NMS
    baked in) and ``config.json`` (inference parameters). Inference runs the
    ONNX graph with ``onnxruntime`` and returns word bounding boxes. No
    ``ultralytics`` dependency (ADR 006).
    """

    HF_REPO_ID = "PellelNitram/xournalpp-htr-word-detector-yolo"

    def __init__(self, session: ort.InferenceSession, config: dict, revision: str):
        super().__init__(revision)
        self.session = session
        self.config = config
        self._input_name = session.get_inputs()[0].name
        self._imgsz = config.get("imgsz", 1024)
        self._conf = config.get("conf", 0.25)

    @classmethod
    def from_pretrained(cls, revision: str = "main") -> "YOLOWordDetectorModel":
        onnx_path = hf_hub_download(cls.HF_REPO_ID, "model.onnx", revision=revision)
        config_path = hf_hub_download(cls.HF_REPO_ID, "config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)
        return cls(
            session=ort.InferenceSession(onnx_path),
            config=config,
            revision=revision,
        )

    @staticmethod
    def _letterbox(
        image: np.ndarray, target_size: int
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize with aspect ratio preserved, pad to square."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        top = pad_h // 2
        left = pad_w // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, (top, left)

    NMS_IOU_THRESHOLD = 0.5

    def detect(
        self, image_grayscale: np.ndarray, conf: float | None = None
    ) -> List[BoundingBox]:
        """Detect word bounding boxes in a grayscale image.

        The returned boxes are in the pixel coordinate system of the
        *passed* image.
        """
        if conf is None:
            conf = self._conf

        if len(image_grayscale.shape) == 2:
            img_rgb = cv2.cvtColor(image_grayscale, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image_grayscale

        orig_h, orig_w = img_rgb.shape[:2]
        padded, scale, (pad_top, pad_left) = self._letterbox(img_rgb, self._imgsz)

        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

        outputs = self.session.run(None, {self._input_name: blob})
        # Raw YOLOv8 ONNX output: (1, 4+num_classes, num_anchors).
        # Transpose to (num_anchors, 4+num_classes).
        preds = outputs[0][0].T

        # Columns: [cx, cy, w, h, class_scores...]
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        scores = preds[:, 4:].max(axis=1)

        mask = scores > conf
        cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]

        # Convert centre-format to corner-format for NMS.
        nms_boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, scores.tolist(), conf, self.NMS_IOU_THRESHOLD
        )
        if len(indices) == 0:
            return []

        boxes = []
        for i in indices.flatten():
            x1 = (cx[i] - w[i] / 2 - pad_left) / scale
            y1 = (cy[i] - h[i] / 2 - pad_top) / scale
            x2 = (cx[i] + w[i] / 2 - pad_left) / scale
            y2 = (cy[i] + h[i] / 2 - pad_top) / scale
            x1 = max(0.0, min(float(orig_w), float(x1)))
            y1 = max(0.0, min(float(orig_h), float(y1)))
            x2 = max(0.0, min(float(orig_w), float(x2)))
            y2 = max(0.0, min(float(orig_h), float(y2)))
            if x2 > x1 and y2 > y1:
                boxes.append(BoundingBox(x1, y1, x2, y2))
        return boxes


class RFDETRWordDetectorModel(HFHubInferenceModel):
    """RF-DETR-based word-detection model, loaded from HF Hub as ONNX.

    The repository contains ``model.onnx`` (exported from a fine-tuned RF-DETR
    checkpoint) and ``config.json`` (inference parameters). Inference runs the
    ONNX graph with ``onnxruntime`` and returns word bounding boxes. No
    ``rfdetr`` dependency (ADR 006).

    Unlike :class:`YOLOWordDetectorModel` this model is NMS-free: RF-DETR is a
    DETR-style detector whose Hungarian matching makes each of its 300 object
    queries responsible for at most one object, so duplicate suppression is
    unnecessary. Decoding mirrors ``rfdetr.models.postprocess.PostProcess``.
    """

    HF_REPO_ID = "PellelNitram/xournalpp-htr-word-detector-rf-detr"

    #: RF-DETR normalises with the ImageNet statistics its backbone was
    #: pretrained on.
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, session: ort.InferenceSession, config: dict, revision: str):
        super().__init__(revision)
        self.session = session
        self.config = config
        self._input_name = session.get_inputs()[0].name
        self._resolution = config.get("resolution", 1024)
        self._threshold = config.get("threshold", 0.5)
        self._word_class_index = config.get("word_class_index", 0)
        self._num_select = config.get("num_select", 300)

    @classmethod
    def from_pretrained(cls, revision: str = "main") -> "RFDETRWordDetectorModel":
        onnx_path = hf_hub_download(cls.HF_REPO_ID, "model.onnx", revision=revision)
        config_path = hf_hub_download(cls.HF_REPO_ID, "config.json", revision=revision)
        with open(config_path) as f:
            config = json.load(f)
        return cls(
            session=ort.InferenceSession(onnx_path),
            config=config,
            revision=revision,
        )

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize to the square model input and normalise.

        The resize deliberately does **not** preserve aspect ratio: RF-DETR
        predicts boxes normalised to the resized frame, and its post-processor
        maps them back by multiplying with the original width/height. Adding
        letterbox padding here would break that mapping.
        """
        resized = cv2.resize(
            image_rgb,
            (self._resolution, self._resolution),
            interpolation=cv2.INTER_LINEAR,
        )
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return np.transpose(blob, (2, 0, 1))[None, :, :, :]

    def detect(
        self, image_grayscale: np.ndarray, threshold: float | None = None
    ) -> List[BoundingBox]:
        """Detect word bounding boxes in a grayscale image.

        The returned boxes are in the pixel coordinate system of the
        *passed* image.
        """
        if threshold is None:
            threshold = self._threshold

        if len(image_grayscale.shape) == 2:
            img_rgb = cv2.cvtColor(image_grayscale, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image_grayscale

        orig_h, orig_w = img_rgb.shape[:2]
        outputs = self.session.run(None, {self._input_name: self._preprocess(img_rgb)})

        # dets: (1, num_queries, 4) normalised cxcywh.
        # labels: (1, num_queries, num_classes) raw logits.
        dets = outputs[0][0]
        logits = outputs[1][0]

        # Select the top scoring query/class pairs, exactly as RF-DETR's
        # PostProcess does: sigmoid, flatten over (query, class), take top-k.
        probs = 1.0 / (1.0 + np.exp(-logits))
        flat = probs.reshape(-1)
        num_classes = probs.shape[1]
        num_select = min(self._num_select, flat.shape[0])
        # Descending score, ties broken by ascending index, matching PostProcess.
        order = np.argsort(-flat, kind="stable")[:num_select]

        boxes: List[BoundingBox] = []
        for idx in order:
            score = float(flat[idx])
            if score <= threshold:
                break  # `order` is sorted, so everything after is lower too.
            if idx % num_classes != self._word_class_index:
                continue
            cx, cy, w, h = dets[idx // num_classes]
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h
            x1 = max(0.0, min(float(orig_w), float(x1)))
            y1 = max(0.0, min(float(orig_h), float(y1)))
            x2 = max(0.0, min(float(orig_w), float(x2)))
            y2 = max(0.0, min(float(orig_h), float(y2)))
            if x2 > x1 and y2 > y1:
                boxes.append(BoundingBox(x1, y1, x2, y2))
        return boxes


class OnlineWordDetectorModel(HFHubInferenceModel):
    """Stroke-based word detector: groups the strokes of a page into words.

    Unlike the other detectors it works on the pen strokes, not on a rendering
    of the page. The repository contains ``model.npz`` (the boosted trees as
    plain arrays) and ``config.json`` (merge threshold and training setup). A
    boosted model scores each pair of consecutive strokes as "same word" or
    not, and the stroke sequence is cut where that score falls below the
    threshold. Evaluation is plain numpy, no scikit-learn or onnxruntime. This
    model performs word *detection* only -- it produces no transcription.
    """

    HF_REPO_ID = "PellelNitram/xournalpp-htr-word-detector-online"

    def __init__(self, ensemble: TreeEnsemble, config: dict, revision: str):
        super().__init__(revision)
        self.ensemble = ensemble
        self.config = config
        self.threshold = float(config["threshold"])

    @classmethod
    def from_pretrained(cls, revision: str = "main") -> "OnlineWordDetectorModel":
        model_path = hf_hub_download(cls.HF_REPO_ID, "model.npz", revision=revision)
        config_path = hf_hub_download(cls.HF_REPO_ID, "config.json", revision=revision)
        return cls._from_files(Path(model_path), Path(config_path), revision)

    @classmethod
    def from_local(
        cls, model_dir: Path, revision: str = "local"
    ) -> "OnlineWordDetectorModel":
        """Load ``model.npz`` and ``config.json`` from a local export directory."""
        model_dir = Path(model_dir)
        return cls._from_files(
            model_dir / "model.npz", model_dir / "config.json", revision
        )

    @classmethod
    def _from_files(
        cls, model_path: Path, config_path: Path, revision: str
    ) -> "OnlineWordDetectorModel":
        with open(config_path) as f:
            config = json.load(f)
        return cls(TreeEnsemble.load(model_path), config, revision)

    def group_strokes(
        self, strokes: List[tuple[np.ndarray, np.ndarray]]
    ) -> List[frozenset[int]]:
        """Group strokes into words.

        :param strokes: ``(x, y)`` coordinate arrays, one per stroke, in writing
            order.
        :returns: Disjoint sets of stroke indices covering all strokes, one per
            detected word.
        """
        if not strokes:
            return []
        if len(strokes) == 1:
            return [frozenset([0])]
        page = StrokePage("", 0, 0, list(strokes), np.full(len(strokes), NON_TEXT))
        proba = self.ensemble.predict_proba(consecutive_pair_features(page))
        return groups_from_merges(len(strokes), proba >= self.threshold)
