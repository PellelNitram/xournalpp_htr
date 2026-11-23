"""Local torch inference from a ``.pth`` checkpoint.

Runs the *trained checkpoint* directly (before/without an ONNX export).
Production inference uses the ONNX path.
"""

import json
from pathlib import Path

import torch

from xournalpp_htr.training.online_htr.config import (
    load_feature_config,
    load_model_config,
)
from xournalpp_htr.training.online_htr.features import Stroke, strokes_to_features
from xournalpp_htr.training.online_htr.network import OnlineHTRNet, greedy_decode
from xournalpp_htr.training.online_htr.utils import get_device


def load_charset(model_path: Path) -> list[str]:
    with open(model_path.parent / "charset.json") as f:
        return json.load(f)


def run_strokes_through_network(
    strokes: list[Stroke],
    model_path: Path = Path("best_model.pth"),
    device: str = "auto",
) -> dict:
    device = get_device(device)
    charset = load_charset(model_path)
    model_cfg = load_model_config(model_path.parent)
    feature_cfg = load_feature_config(model_path.parent)

    net = OnlineHTRNet(num_classes=len(charset) + 1, cfg=model_cfg)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device).eval()

    features = strokes_to_features(strokes, feature_cfg.points_per_unit)
    tensor_input = torch.from_numpy(features[:, None, :]).to(device)
    with torch.no_grad():
        log_probs = net(tensor_input)

    return {
        "text": greedy_decode(log_probs, charset)[0],
        "features": features,
        "log_probs": log_probs[:, 0, :].cpu().numpy(),
        "charset": charset,
    }
