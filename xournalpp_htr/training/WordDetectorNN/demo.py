from pathlib import Path
import cv2
import argparse

import torch
import gradio as gr
import numpy as np

from my_code import draw_bboxes_on_image
from my_code import run_image_through_network
from my_code import get_example_list


parser = argparse.ArgumentParser(
        description="Train a WordDetectorNet model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
parser.add_argument("--model_path", type=Path, default=Path('best_model.pth'),
                    help="Path to trained pth model.")
parser.add_argument("--device", type=str, choices=["cpu", "auto"], default="cpu",
                    help="Selects the device. \"auto\" selects GPU if available.")
args = vars(parser.parse_args())

model_path = args['model_path']
device_selection = args['device']

print(f'Used args: {args}')

def process_image(
        image: np.ndarray, # Is (H, W, 3) uint8 RGB; return needs to be the same
        margin: float,
    ) -> np.ndarray:

    margin = int(margin)

    image_BGR = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

    if device_selection == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    # Inference
    result = run_image_through_network(
        image_grayscale=image_gray,
        model_path=model_path,
        device=device,
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
    title="WordDetectorNN: Handwritten Word Detection",
    description="Detect handwritten words in your images. Upload an image of handwritten text, adjust the margin slider (start at 0) and see bounding boxes appear around each detected word.",
    article="""
    ### About this project
    This demo is part of **[Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr)**, an effort to bring handwritten text recognition to [Xournal++](https://github.com/xournalpp/xournalpp).

    The original WordDetectorNN model was invented and implemented by **Harald Scheidl** in his [WordDetectorNN repository](https://github.com/githubharald/WordDetectorNN).  
    I’ve re-implemented it with some PyTorch best practices and shared it here on Hugging Face. Thanks Harald for the great implementation!

    Hope everyone enjoys experimenting with it! 🙂
    """,
    examples=get_example_list(),
    cache_examples=True,
)

demo.launch()
