"""Export a trained OnlineHTR checkpoint to the ONNX inference artifact.

ADR 006: ONNX is the canonical inference format. Produces

* ``model.onnx``  -- the network graph (input ``ink`` ``(T, N, 3)``, output
  ``log_probs`` ``(T, N, num_classes)``, CTC blank last). Inference runs one
  sample at a time (N = 1): the legacy exporter warns that LSTM graphs
  exported this way may not generalise to other batch sizes.
* ``config.json`` -- charset and feature parameters.

Usage::

    uv run --extra training-online-htr python -m \\
        xournalpp_htr.training.online_htr.export \\
        --checkpoint experiments/experiment1/baseline/best_model.pth \\
        --output-dir experiments/experiment1/baseline/export
"""

import argparse
import json
from pathlib import Path

import torch

from xournalpp_htr.training.online_htr.config import (
    FeatureConfig,
    ModelConfig,
    load_feature_config,
    load_model_config,
)
from xournalpp_htr.training.online_htr.features import NUM_CHANNELS
from xournalpp_htr.training.online_htr.infer import load_charset
from xournalpp_htr.training.online_htr.network import OnlineHTRNet

# Not published yet; only used by `--upload`.
HF_REPO_ID = "PellelNitram/xournalpp-htr-online-htr"


def build_config(
    charset: list[str], model_cfg: ModelConfig, feature_cfg: FeatureConfig
) -> dict:
    return {
        "model_name": "online_htr",
        "charset": charset,
        "num_classes": len(charset) + 1,
        "features": {
            "channels": ["dx", "dy", "n"],
            "points_per_unit": feature_cfg.points_per_unit,
        },
        "input_channels": model_cfg.input_channels,
    }


def export(checkpoint: Path, output_dir: Path) -> dict:
    """Export ``checkpoint`` to ``output_dir`` as ``model.onnx`` + ``config.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    charset = load_charset(checkpoint)
    model_cfg = load_model_config(checkpoint.parent)
    feature_cfg = load_feature_config(checkpoint.parent)
    assert model_cfg.input_channels == NUM_CHANNELS

    net = OnlineHTRNet(num_classes=len(charset) + 1, cfg=model_cfg)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.eval()

    dummy_input = torch.zeros(50, 1, model_cfg.input_channels, dtype=torch.float32)
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        str(onnx_path),
        input_names=["ink"],
        output_names=["log_probs"],
        dynamic_axes={
            "ink": {0: "time", 1: "batch"},
            "log_probs": {0: "time", 1: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(build_config(charset, model_cfg, feature_cfg), f, indent=2)

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
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("exports"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    export(args.checkpoint, args.output_dir)
    if args.upload:
        upload_to_hub(args.output_dir)


if __name__ == "__main__":
    main()
