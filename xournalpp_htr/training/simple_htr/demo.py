"""Local Gradio demo for a trained SimpleHTR checkpoint.

An interactive Gradio app to sanity-check whether a trained checkpoint
recognises handwritten words. Per ADR 007 it runs **locally**
(``demo.launch()``); it is not deployed as a HuggingFace Space.

    uv run python -m xournalpp_htr.training.simple_htr.demo --help
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from xournalpp_htr.training.simple_htr.infer import run_image_through_network


def build_demo(model_path: Path, device_selection: str) -> gr.Blocks:
    def process_image(image: np.ndarray) -> str:
        if image is None:
            return ""
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        result = run_image_through_network(
            image_grayscale=image_gray,
            model_path=model_path,
            device=device_selection,
        )
        return result["text"]

    def process_sketch(sketch: dict) -> str:
        if sketch is None:
            return ""
        composite = sketch["composite"]
        if composite is None:
            return ""
        return process_image(composite)

    article = """
    ### About this project
    Part of **[Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr)**,
    an open-source effort to bring handwritten text recognition to
    [Xournal++](https://github.com/xournalpp/xournalpp).

    The original SimpleHTR model was created by **Harald Scheidl**
    ([SimpleHTR](https://github.com/githubharald/SimpleHTR)) and is
    reimplemented here with PyTorch. Thanks Harald!
    """

    with gr.Blocks(title="SimpleHTR: Handwritten Word Recognition (local)") as demo:
        gr.Markdown("# SimpleHTR: Handwritten Word Recognition (local)")
        gr.Markdown("Recognise a single handwritten word. Upload an image or draw one.")

        with gr.Tabs():
            with gr.TabItem("Upload image"):
                upload_input = gr.Image(type="numpy", label="Word image")
                upload_output = gr.Textbox(label="Recognised text")
                upload_btn = gr.Button("Recognise")
                upload_btn.click(
                    process_image, inputs=upload_input, outputs=upload_output
                )

            with gr.TabItem("Draw"):
                sketch_input = gr.Sketchpad(
                    label="Draw a word",
                    brush=gr.Brush(default_size=3, colors=["black"]),
                    canvas_size=(400, 100),
                )
                sketch_output = gr.Textbox(label="Recognised text")
                sketch_btn = gr.Button("Recognise")
                sketch_btn.click(
                    process_sketch, inputs=sketch_input, outputs=sketch_output
                )

        gr.Markdown(article)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("best_model.pth"),
        help="Path to the trained .pth checkpoint.",
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
    args = parser.parse_args()

    demo = build_demo(args.model_path, args.device)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
