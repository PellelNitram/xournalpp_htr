"""Export a trained WordDetector checkpoint to the ONNX inference artifact.

ADR 006: ONNX is the canonical inference format. This script produces the two
files consumed by :class:`xournalpp_htr.inference_models.WordDetectorModel`:

* ``model.onnx``  -- the network with softmax baked into the graph.
* ``config.json`` -- the pre/post-processing parameters.

Uploading the export to HF Hub (``PellelNitram/xournalpp-htr-word-detector``)
is a separate, authenticated step (see :func:`upload_to_hub`); it is *not* run
automatically and is not exercised by the test suite.

Usage::

    uv run python -m xournalpp_htr.training.word_detector.export \\
        --checkpoint best_model.pth --output-dir exports/
"""

import argparse
import json
from pathlib import Path

import torch

from xournalpp_htr.training.shared.postprocessing import MapOrdering
from xournalpp_htr.training.word_detector.config import (
    DetectionConfig,
    NormalizationConfig,
)
from xournalpp_htr.training.word_detector.network import WordDetectorNet

_DETECTION_DEFAULTS = DetectionConfig()
_NORMALIZATION_DEFAULTS = NormalizationConfig()

HF_REPO_ID = "PellelNitram/xournalpp-htr-word-detector"


class _SoftmaxBaked(torch.nn.Module):
    """Wrapper that bakes ``apply_softmax=True`` into the traced graph."""

    def __init__(self, net: WordDetectorNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x, apply_softmax=True)


def build_config() -> dict:
    """Pre/post-processing parameters stored alongside ``model.onnx``."""
    return {
        "model_name": "word_detector",
        "input_size": {
            "height": WordDetectorNet.input_size[0],
            "width": WordDetectorNet.input_size[1],
        },
        "output_size": {
            "height": WordDetectorNet.output_size[0],
            "width": WordDetectorNet.output_size[1],
        },
        "fg_cc_threshold": _DETECTION_DEFAULTS.fg_threshold,
        "fg_cc_max_num": _DETECTION_DEFAULTS.max_detections,
        # Image normalisation: (pixel / scale) + shift  -> see
        # shared.postprocessing.normalize_image_transform.
        "normalization": {
            "scale": _NORMALIZATION_DEFAULTS.scale,
            "shift": _NORMALIZATION_DEFAULTS.shift,
        },
        "map_ordering": {
            "SEG_WORD": MapOrdering.SEG_WORD,
            "SEG_SURROUNDING": MapOrdering.SEG_SURROUNDING,
            "SEG_BACKGROUND": MapOrdering.SEG_BACKGROUND,
            "GEO_TOP": MapOrdering.GEO_TOP,
            "GEO_BOTTOM": MapOrdering.GEO_BOTTOM,
            "GEO_LEFT": MapOrdering.GEO_LEFT,
            "GEO_RIGHT": MapOrdering.GEO_RIGHT,
            "NUM_MAPS": MapOrdering.NUM_MAPS,
        },
    }


def export(checkpoint: Path, output_dir: Path) -> dict:
    """Export ``checkpoint`` to ``output_dir`` as ``model.onnx`` + ``config.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    net = WordDetectorNet()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.eval()

    model = _SoftmaxBaked(net).eval()
    h, w = WordDetectorNet.input_size
    dummy_input = torch.zeros(1, 1, h, w, dtype=torch.float32)

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["image"],
        output_names=["maps"],
        dynamic_axes={"image": {0: "batch"}, "maps": {0: "batch"}},
        opset_version=17,
        # WordDetectorNet has no data-dependent control flow, so the stable
        # legacy TorchScript exporter traces it cleanly and avoids the
        # onnxscript dependency the dynamo path pulls in (ADR 006).
        dynamo=False,
    )

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(build_config(), f, indent=2)

    print(f"Wrote {onnx_path} and {config_path}")
    return {"onnx": onnx_path, "config": config_path}


def upload_to_hub(output_dir: Path, repo_id: str = HF_REPO_ID) -> None:
    """Upload ``model.onnx`` + ``config.json`` to HF Hub (ADR 006 section 1).

    Requires HF authentication (``huggingface-cli login`` or ``HF_TOKEN``) and
    write access to ``repo_id``. Run manually after :func:`export`.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    for filename in ("model.onnx", "config.json"):
        api.upload_file(
            path_or_fileobj=str(output_dir / filename),
            path_in_repo=filename,
            repo_id=repo_id,
        )
    print(f"Uploaded model.onnx + config.json to {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to the trained .pth checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Directory to write model.onnx and config.json into.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After export, upload to HF Hub (requires authentication).",
    )
    args = parser.parse_args()

    export(args.checkpoint, args.output_dir)
    if args.upload:
        upload_to_hub(args.output_dir)


if __name__ == "__main__":
    main()
