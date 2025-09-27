import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from my_code import (
    draw_bboxes_on_image,
    get_example_list,
    run_image_through_network,
    save_event,
)

load_dotenv()

DEMO = os.getenv("DEMO") == "1"
SB_URL = os.getenv("SB_URL")
SB_KEY = os.getenv("SB_KEY")
SB_BUCKET_NAME = os.getenv("SB_BUCKET_NAME")
SB_SCHEMA_NAME = os.getenv("SB_SCHEMA_NAME")
SB_TABLE_NAME = os.getenv("SB_TABLE_NAME")

parser = argparse.ArgumentParser(
    description="Train a WordDetectorNet model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model_path",
    type=Path,
    default=Path("best_model.pth"),
    help="Path to trained pth model.",
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "auto"],
    default="cpu",
    help='Selects the device. "auto" selects GPU if available.',
)
args = vars(parser.parse_args())

model_path = args["model_path"]
device_selection = args["device"]

print(f"Used args: {args}")


def process_image(
    image: np.ndarray,  # Is (H, W, 3) uint8 RGB; return needs to be the same
    margin: float,
    donate_data: bool,
) -> np.ndarray:
    margin = int(margin)

    image_BGR = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

    if device_selection == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    # Inference
    result = run_image_through_network(
        image_grayscale=image_gray,
        model_path=model_path,
        device=device,
    )

    # Post processing
    scaling_factors = np.array(image_gray.shape) / np.array(
        result["model_input_image"].shape
    )
    bboxes_scaled = [
        aabb.scale(*scaling_factors[::-1]).enlarge(margin_x=margin, margin_y=margin)
        for aabb in result["aabbs"]
    ]
    vis_scaled = draw_bboxes_on_image(image_BGR, bboxes_scaled, denormalise=False)

    save_event(
        {
            "timestamp": datetime.now(timezone.utc),
            "demo": DEMO,
            "donate_data": donate_data,
            "uuid": uuid4(),
            "image": image,
        },
        SB_URL=SB_URL,
        SB_KEY=SB_KEY,
        SB_SCHEMA_NAME=SB_SCHEMA_NAME,
        SB_TABLE_NAME=SB_TABLE_NAME,
        SB_BUCKET_NAME=SB_BUCKET_NAME,
    )

    return cv2.cvtColor(vis_scaled, cv2.COLOR_BGR2RGB)


demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Input image."),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            step=1,
            label="Margin",
        ),
        gr.Checkbox(
            value=False,
            label="Donate Data",
            info="By checking this box, you agree to share your uploaded image to help improve our open-source models. Donated data will be open source and freely available as dataset.",
        ),
    ],
    outputs=gr.Image(
        type="numpy", label="Input image with detected words superimposed."
    ),
    title="WordDetectorNN: Handwritten Word Detection",
    description="Detect handwritten words in images. Upload an image, adjust the margin slider, and see bounding boxes around detected words.",
    article="""
    ### About this project
    This demo is part of **[Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr)**, an open-source effort to bring handwritten text recognition to [Xournal++](https://github.com/xournalpp/xournalpp).

    The original WordDetectorNN model was created by **Harald Scheidl** in his [WordDetectorNN repository](https://github.com/githubharald/WordDetectorNN). This project re-implements it with PyTorch best practices. Thanks Harald for the great work and inspiration!

    Donated data will contribute to an open-source dataset for the community. Thank you for supporting open-source innovation!
    """,
    examples=get_example_list(),
    cache_examples=True,
)

demo.launch()
