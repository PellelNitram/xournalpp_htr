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
