"""Local Gradio demo for a trained WordDetector checkpoint.

An interactive Gradio app to sanity-check whether a trained checkpoint detects
words. Per ADR 007 it runs **locally** (``demo.launch()``); it is not deployed
as a HuggingFace Space and has no telemetry / data-donation / Supabase.

    uv run python -m xournalpp_htr.training.word_detector.demo --help
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from xournalpp_htr.training.shared.postprocessing import draw_bboxes_on_image
from xournalpp_htr.training.word_detector.infer import run_image_through_network
from xournalpp_htr.training.word_detector.utils import get_device, get_example_list


def build_demo(model_path: Path, device_selection: str) -> gr.Interface:
    def process_image(image: np.ndarray, margin: float) -> np.ndarray:
        """``image`` is (H, W, 3) uint8 RGB; the return has the same shape."""
        margin = int(margin)

        image_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        device = get_device(device_selection)

        result = run_image_through_network(
            image_grayscale=image_gray,
            model_path=model_path,
            device=device,
        )

        # Scale boxes from the fixed network-input size back to the input image.
        scaling_factors = np.array(image_gray.shape) / np.array(
            result["model_input_image"].shape
        )
        bboxes_scaled = [
            aabb.scale(*scaling_factors[::-1]).enlarge(margin_x=margin, margin_y=margin)
            for aabb in result["aabbs"]
        ]
        vis = draw_bboxes_on_image(image_bgr, bboxes_scaled, denormalise=False)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="numpy", label="Input image."),
            gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Margin"),
        ],
        outputs=gr.Image(
            type="numpy", label="Input image with detected words superimposed."
        ),
        title="WordDetectorNN: Handwritten Word Detection (local)",
        description="Detect handwritten words in an image. Upload an image, "
        "adjust the margin slider, and see the predicted bounding boxes.",
        article="""
        ### About this project
        Part of **[Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr)**,
        an open-source effort to bring handwritten text recognition to
        [Xournal++](https://github.com/xournalpp/xournalpp).

        The original WordDetectorNN model was created by **Harald Scheidl**
        ([WordDetectorNN](https://github.com/githubharald/WordDetectorNN)) and
        is reimplemented here with PyTorch best practices. Thanks Harald!
        """,
        examples=[[url, 0] for url in get_example_list()],
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
