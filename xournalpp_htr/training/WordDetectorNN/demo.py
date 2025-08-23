from pathlib import Path
import cv2
import argparse

import torch
import gradio as gr
import numpy as np

from my_code import draw_bboxes_on_image
from my_code import run_image_through_network


parser = argparse.ArgumentParser(
        description="Train a WordDetectorNet model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
parser.add_argument("--model_path", type=Path, required=True,
                    help="Path to trained model.")
args = vars(parser.parse_args())

model_path = args['model_path']

def process_image(
        image: np.ndarray, # Is (H, W, 3) uint8 RGB; return needs to be the same
        margin: float,
    ) -> np.ndarray:

    margin = int(margin)

    image_BGR = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

    # Inference
    result = run_image_through_network(
        image_grayscale=image_gray,
        model_path=model_path,
    )

    # Post processing
    scaling_factors = np.array(image_gray.shape) / np.array(result['model_input_image'].shape)
    bboxes_scaled = [ aabb.scale(*scaling_factors[::-1]).enlarge(margin_x=margin, margin_y=margin) for aabb in result['aabbs'] ]
    vis_scaled = draw_bboxes_on_image(image_BGR, bboxes_scaled, denormalise=False)

    return cv2.cvtColor(vis_scaled, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Input image."),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=0, # Default value
            step=1,
            label="Margin"
        )
    ],
    outputs=gr.Image(type="numpy", label="Input image with detected words superimposed."),
    title="WordDetectorNN",
    description="Upload an image of handwritten text. Adjust the margin to add additional margin to the detected bounding boxes."
)

demo.launch()
