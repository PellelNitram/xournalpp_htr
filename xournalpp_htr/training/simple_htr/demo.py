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


def build_demo(model_path: Path, device_selection: str) -> gr.Interface:
    def process_image(image: np.ndarray) -> str:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        result = run_image_through_network(
            image_grayscale=image_gray,
            model_path=model_path,
            device=device_selection,
        )
        return result["text"]

    return gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="numpy", label="Word image"),
        outputs=gr.Textbox(label="Recognised text"),
        title="SimpleHTR: Handwritten Word Recognition (local)",
        description="Recognise a single handwritten word. Upload an image of "
        "a handwritten word to get the predicted text.",
        article="""
        ### About this project
        Part of **[Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr)**,
        an open-source effort to bring handwritten text recognition to
        [Xournal++](https://github.com/xournalpp/xournalpp).

        The original SimpleHTR model was created by **Harald Scheidl**
        ([SimpleHTR](https://github.com/githubharald/SimpleHTR)) and is
        reimplemented here with PyTorch. Thanks Harald!
        """,
        cache_examples=False,
    )


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
