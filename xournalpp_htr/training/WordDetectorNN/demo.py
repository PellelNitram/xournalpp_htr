import torch

import gradio as gr
import numpy as np

from pathlib import Path
import cv2

from my_code import draw_bboxes_on_image
from my_code import run_image_through_network


def process_image(
        image: np.ndarray, # Is (H, W, 3) uint8 RGB; return needs to be the same
    ) -> np.ndarray:

    image_BGR = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

    # Inference
    result = run_image_through_network(
        image_grayscale=image_gray,
        model_path=Path('best_model.pth'),
    )

    # Post processing
    scaling_factors = np.array(image_gray.shape) / np.array(result['model_input_image'].shape)
    bboxes_scaled = [ aabb.scale(*scaling_factors[::-1]) for aabb in result['aabbs'] ]
    vis_scaled = draw_bboxes_on_image(image_BGR, bboxes_scaled, denormalise=False)

    return cv2.cvtColor(vis_scaled, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=process_image,
    inputs="image",
    outputs="image",
)

demo.launch()
