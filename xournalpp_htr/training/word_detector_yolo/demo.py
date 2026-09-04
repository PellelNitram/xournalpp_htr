"""Local Gradio demo for a trained YOLO word detector checkpoint.

Per ADR 007 it runs **locally**; it is not deployed as a HuggingFace Space.

    uv run python -m xournalpp_htr.training.word_detector_yolo.demo --help
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent


def find_latest_model() -> Path:
    runs_dir = SCRIPT_DIR / "runs"
    candidates = sorted(runs_dir.rglob("train_*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("No trained model found. Run train.py first.")
    return candidates[-1]


def build_demo(model_path: Path, device: str) -> gr.Blocks:
    model = YOLO(str(model_path))
    print(f"Loaded model: {model_path}")

    def predict(image, confidence: float) -> np.ndarray:
        if image is None:
            return None
        if isinstance(image, dict):
            image = image.get("composite", image.get("image"))
        if image is None:
            return None
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        results = model.predict(
            image, conf=confidence, imgsz=1024, device=device, verbose=False
        )
        return results[0].plot()

    with gr.Blocks(title="Handwritten Word Detection (YOLO)") as app:
        gr.Markdown(
            f"## Handwritten Word Detection (YOLO)\n"
            f"Model: `{model_path.parent.parent.name}`"
        )

        conf_slider = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=0.25,
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
                    inputs=[upload_input, conf_slider],
                    outputs=upload_output,
                )

            with gr.TabItem("Draw"):
                with gr.Row():
                    canvas = gr.Sketchpad(type="numpy", label="Draw here")
                    draw_output = gr.Image(label="Detections")
                draw_btn = gr.Button("Detect words")
                draw_btn.click(
                    predict,
                    inputs=[canvas, conf_slider],
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
        help="Path to the trained YOLO .pt checkpoint. Auto-discovers latest if omitted.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help='Inference device. "auto" selects GPU if available.',
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a temporary public Gradio share link.",
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model_path = args.model_path or find_latest_model()
    device = args.device if args.device != "auto" else None

    app = build_demo(model_path, device)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
