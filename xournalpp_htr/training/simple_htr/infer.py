"""Local torch inference from a ``.pth`` checkpoint.

Used by the Gradio demo and the model-inspection notebook to run the *trained
checkpoint* directly (before/without an ONNX export). Production inference uses
the ONNX path via :class:`xournalpp_htr.inference_models.SimpleHTRModel`.
"""

from pathlib import Path

import numpy as np
import torch

from xournalpp_htr.training.simple_htr.dataset import preprocess_image
from xournalpp_htr.training.simple_htr.network import SimpleHTRNet, greedy_decode
from xournalpp_htr.training.simple_htr.utils import get_device


def run_image_through_network(
    image_grayscale: np.ndarray,
    model_path: Path = Path("best_model.pth"),
    device: str = "auto",
) -> dict:
    device = get_device(device)
    model = SimpleHTRNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preprocessed = preprocess_image(
        image_grayscale, SimpleHTRNet.input_height, SimpleHTRNet.input_width
    )
    normalized = preprocessed.astype(np.float32) / 255.0 - 0.5
    tensor_input = torch.from_numpy(normalized[None, None, :, :]).to(device)

    with torch.no_grad():
        log_probs = model(tensor_input)

    decoded = greedy_decode(log_probs)

    return {
        "text": decoded[0],
        "model_input_image": normalized,
    }
