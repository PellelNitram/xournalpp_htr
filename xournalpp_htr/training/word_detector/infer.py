"""Local torch inference from a ``.pth`` checkpoint.

Used by the Gradio demo and the model-inspection notebook to run the *trained
checkpoint* directly (before/without an ONNX export). Production inference uses
the ONNX path via :class:`xournalpp_htr.inference_models.WordDetectorModel`.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from xournalpp_htr.training.shared.bounding_box import BoundingBox
from xournalpp_htr.training.shared.postprocessing import (
    MapOrdering,
    cluster_aabbs,
    decode,
    fg_by_cc,
    normalize_image_transform,
)
from xournalpp_htr.training.word_detector.config import DetectionConfig
from xournalpp_htr.training.word_detector.network import WordDetectorNet
from xournalpp_htr.training.word_detector.utils import get_device

_DETECTION_DEFAULTS = DetectionConfig()


def run_image_through_network(
    image_grayscale: np.ndarray,
    model_path: Path = Path("best_model.pth"),
    device: str = "auto",
) -> dict:
    device = get_device(device)
    model = WordDetectorNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_gray_rescaled = cv2.resize(image_grayscale, WordDetectorNet.input_size)
    image_transformed, _ = normalize_image_transform(image_gray_rescaled, None)
    image_transformed = image_transformed.astype(np.float32)
    image_transformed = torch.from_numpy(image_transformed[None, None, :, :]).to(device)

    with torch.no_grad():
        output_image = model(image_transformed, apply_softmax=True)

    seg = output_image[:, MapOrdering.SEG_WORD : MapOrdering.SEG_BACKGROUND + 1, :, :]
    assert seg.min() >= 0.0
    assert seg.max() <= 1.0

    output_image = output_image.to("cpu").numpy()[0, :, :, :]

    decoded_aabbs = decode(
        output_image,
        scale=WordDetectorNet.input_size[0] / WordDetectorNet.output_size[0],
        comp_fg=fg_by_cc(
            thres=_DETECTION_DEFAULTS.fg_threshold,
            max_num=_DETECTION_DEFAULTS.max_detections,
        ),
    )
    model_input_image = image_transformed[0, 0, :, :].to("cpu").numpy()
    h, w = model_input_image.shape
    aabbs = [aabb.clip(BoundingBox(0, 0, w - 1, h - 1)) for aabb in decoded_aabbs]
    clustered_aabbs = cluster_aabbs(aabbs)

    return {
        "aabbs": clustered_aabbs,
        "model_input_image": model_input_image,
    }
