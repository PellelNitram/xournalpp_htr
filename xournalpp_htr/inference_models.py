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

import json
from abc import ABC, abstractmethod
from typing import ClassVar, List

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from xournalpp_htr.training.shared.bounding_box import BoundingBox
from xournalpp_htr.training.shared.postprocessing import (
    MapOrdering,
    cluster_aabbs,
    decode,
    fg_by_cc,
    normalize_image_transform,
)


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

        # Correct geometry channels: the network multiplies by 448 (training
        # input size), but the actual input may be a different height.
        geo_correction = net_input.shape[2] / self.config["input_size"]["height"]
        output[MapOrdering.GEO_TOP :] *= geo_correction

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

    def recognize(self, image_grayscale: np.ndarray) -> str:
        """Recognise text in a grayscale word image.

        The image is resized to the network's expected input dimensions
        (uniform scale, centered on white canvas) and normalised before inference.
        """
        input_size = self.config["input_size"]
        in_h, in_w = input_size["height"], input_size["width"]
        norm = self.config["normalization"]

        h, w = image_grayscale.shape[:2]
        scale = min(in_w / w, in_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image_grayscale, (new_w, new_h))

        canvas = np.ones((in_h, in_w), dtype=np.uint8) * 255
        y_off = (in_h - new_h) // 2
        x_off = (in_w - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        normalised = canvas.astype(np.float32) / norm["scale"] + norm["shift"]
        net_input = normalised[None, None, :, :]

        log_probs = self.session.run(None, {self._input_name: net_input})[0]
        # log_probs shape: (seq_len, batch, num_classes)
        predictions = log_probs[:, 0, :].argmax(axis=1)

        blank = len(self._charset)
        chars = []
        prev = blank
        for idx in predictions:
            if idx != prev and idx != blank:
                chars.append(self._charset[idx])
            prev = idx
        return "".join(chars)
