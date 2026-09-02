"""Gradio demo for handwritten word detection.

Usage:
    uv run python demo.py [--port 7860]
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO


def find_latest_model() -> Path:
    runs_dir = Path("runs")
    candidates = sorted(runs_dir.rglob("train_*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("No trained model found. Run train.py first.")
    return candidates[-1]


def predict(image, confidence: float) -> np.ndarray:
    if image is None:
        return None
    if isinstance(image, dict):
        image = image.get("composite", image.get("image"))
    if image is None:
        return None
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    results = MODEL.predict(image, conf=confidence, imgsz=1024, verbose=False)
    return results[0].plot()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Handwritten Word Detection") as app:
        gr.Markdown(
            f"## Handwritten Word Detection\nModel: `{MODEL_PATH.parent.parent.name}`"
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
                    predict, inputs=[upload_input, conf_slider], outputs=upload_output
                )

            with gr.TabItem("Draw"):
                with gr.Row():
                    canvas = gr.Sketchpad(type="numpy", label="Draw here")
                    draw_output = gr.Image(label="Detections")
                draw_btn = gr.Button("Detect words")
                draw_btn.click(
                    predict, inputs=[canvas, conf_slider], outputs=draw_output
                )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    MODEL_PATH = find_latest_model()
    MODEL = YOLO(str(MODEL_PATH))
    print(f"Loaded model: {MODEL_PATH}")

    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=args.port)
