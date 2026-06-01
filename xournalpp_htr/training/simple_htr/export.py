"""Export a trained SimpleHTR checkpoint to the ONNX inference artifact.

ADR 006: ONNX is the canonical inference format. This script produces the two
files consumed by :class:`xournalpp_htr.inference_models.SimpleHTRModel`:

* ``model.onnx``  -- the network graph.
* ``config.json`` -- charset, input dimensions, and pre-processing parameters.

Usage::

    uv run python -m xournalpp_htr.training.simple_htr.export \\
        --checkpoint best_model.pth --output-dir exports/
"""

import argparse
import json
from pathlib import Path

import torch

from xournalpp_htr.training.simple_htr.dataset import CHARSET
from xournalpp_htr.training.simple_htr.network import SimpleHTRNet

HF_REPO_ID = "PellelNitram/xournalpp-htr-simple-htr"


def build_config() -> dict:
    return {
        "model_name": "simple_htr",
        "input_size": {
            "height": SimpleHTRNet.input_height,
            "width": SimpleHTRNet.input_width,
        },
        "charset": CHARSET,
        "num_classes": SimpleHTRNet.num_classes,
        "normalization": {
            "scale": 255.0,
            "shift": -0.5,
        },
    }


def export(checkpoint: Path, output_dir: Path) -> dict:
    """Export ``checkpoint`` to ``output_dir`` as ``model.onnx`` + ``config.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    net = SimpleHTRNet()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.eval()

    h, w = SimpleHTRNet.input_height, SimpleHTRNet.input_width
    dummy_input = torch.zeros(1, 1, h, w, dtype=torch.float32)

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        str(onnx_path),
        input_names=["image"],
        output_names=["log_probs"],
        dynamic_axes={"image": {0: "batch"}, "log_probs": {1: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(build_config(), f, indent=2)

    print(f"Wrote {onnx_path} and {config_path}")
    return {"onnx": onnx_path, "config": config_path}


def upload_to_hub(output_dir: Path, repo_id: str = HF_REPO_ID) -> None:
    """Upload ``model.onnx`` + ``config.json`` to HF Hub (ADR 006 section 1)."""
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
