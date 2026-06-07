"""Local torch inference from a ``.pth`` checkpoint.

Used by the Gradio demo and the model-inspection notebook to run the *trained
checkpoint* directly (before/without an ONNX export). Production inference uses
the ONNX path via :class:`xournalpp_htr.inference_models.SimpleHTRModel`.
"""

import json
from pathlib import Path

import numpy as np
import torch

from xournalpp_htr.training.simple_htr.config import load_model_config
from xournalpp_htr.training.simple_htr.dataset import preprocess_image
from xournalpp_htr.training.simple_htr.network import SimpleHTRNet, greedy_decode
from xournalpp_htr.training.simple_htr.utils import get_device


def load_charset(model_path: Path) -> list[str]:
    charset_path = model_path.parent / "charset.json"
    with open(charset_path) as f:
        return json.load(f)


def run_image_through_network(
    image_grayscale: np.ndarray,
    model_path: Path = Path("best_model.pth"),
    device: str = "auto",
) -> dict:
    device = get_device(device)
    charset = load_charset(model_path)
    num_classes = len(charset) + 1
    model_cfg = load_model_config(model_path.parent)

    model = SimpleHTRNet(num_classes=num_classes, cfg=model_cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preprocessed = preprocess_image(
        image_grayscale, model_cfg.input_height, model_cfg.input_width
    )
    normalized = preprocessed.astype(np.float32) / 255.0 - 0.5
    tensor_input = torch.from_numpy(normalized[None, None, :, :]).to(device)

    with torch.no_grad():
        log_probs = model(tensor_input)

    decoded = greedy_decode(log_probs, charset)

    return {
        "text": decoded[0],
        "model_input_image": normalized,
        "log_probs": log_probs[:, 0, :].cpu().numpy(),
        "charset": charset,
    }
