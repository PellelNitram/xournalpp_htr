"""Local Gradio demo for a trained RF-DETR word detector checkpoint.

Per ADR 007 it runs **locally**; it is not deployed as a HuggingFace Space.

    uv run python -m xournalpp_htr.training.word_detector_rf_detr.demo --help
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from xournalpp_htr.training.word_detector_rf_detr.config import (
    InferenceConfig,
    ModelConfig,
)
from xournalpp_htr.training.word_detector_rf_detr.predict import (
    draw_predictions,
    load_model,
)

SCRIPT_DIR = Path(__file__).resolve().parent
_INFERENCE_DEFAULTS = InferenceConfig()
_MODEL_DEFAULTS = ModelConfig()

CHECKPOINT_NAME = "checkpoint_best_total.pth"


def find_latest_model() -> Path:
    outputs_dir = SCRIPT_DIR / "outputs"
    candidates = sorted(outputs_dir.rglob(f"train_*/{CHECKPOINT_NAME}"))
    if not candidates:
        raise FileNotFoundError("No trained model found. Run train.py first.")
    return candidates[-1]


def build_demo(model_path: Path, variant: str, resolution: int) -> gr.Blocks:
    model = load_model(model_path, variant, resolution)
    print(f"Loaded model: {model_path}")

    def predict(image, threshold: float) -> np.ndarray | None:
        if image is None:
            return None
        if isinstance(image, dict):
            image = image.get("composite", image.get("image"))
        if image is None:
            return None
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # Gradio hands us RGB, which is what RF-DETR wants.
        detections = model.predict(image, threshold=threshold)
        return draw_predictions(image, detections)

    with gr.Blocks(title="Handwritten Word Detection (RF-DETR)") as app:
        gr.Markdown(
            f"## Handwritten Word Detection (RF-DETR)\n"
            f"Model: `{model_path.parent.name}`"
        )

        threshold_slider = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=_INFERENCE_DEFAULTS.threshold,
            step=0.05,
            label="Confidence threshold",
        )

        with gr.Tabs():
            with gr.TabItem("Upload image"):
                with gr.Row():
                    upload_input = gr.Image(type="numpy", label="Upload")
                    upload_output = gr.Image(label="Detections")
                upload_btn = gr.Button("Detect words")
                upload_btn.click(
                    predict,
                    inputs=[upload_input, threshold_slider],
                    outputs=upload_output,
                )

            with gr.TabItem("Draw"):
                with gr.Row():
                    canvas = gr.Sketchpad(type="numpy", label="Draw here")
                    draw_output = gr.Image(label="Detections")
                draw_btn = gr.Button("Detect words")
                draw_btn.click(
                    predict,
                    inputs=[canvas, threshold_slider],
                    outputs=draw_output,
                )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the trained RF-DETR .pth checkpoint. Auto-discovers latest if omitted.",
    )
    parser.add_argument("--variant", type=str, default=_MODEL_DEFAULTS.variant)
    parser.add_argument(
        "--resolution", type=int, default=_INFERENCE_DEFAULTS.resolution
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a temporary public Gradio share link.",
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model_path = args.model_path or find_latest_model()

    app = build_demo(model_path, args.variant, args.resolution)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
