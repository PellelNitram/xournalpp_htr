import torch

import gradio as gr
import numpy as np

from pathlib import Path
from my_code import WordDetectorNet
import cv2


# ========
# Settings
# ========

model_path = Path('best_model.pth') # later, replace w/ cli argument

# ================
# Configure system
# ================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========
# Load model
# ==========

model = WordDetectorNet()  # instantiate your model
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# =========
# Gradio UI
# =========

def process_image(image):
    print('type(image):', type(image))
    print('image.shape:', image.shape)
    print('image.dtype:', image.dtype)
    # Save the original RGB image
    # np.save('uploaded_image_rgb.npy', image)
    # Convert RGB to grayscale using OpenCV
    # grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Save the grayscale image
    # np.save('uploaded_image_grayscale.npy', grayscale)
    return image

demo = gr.Interface(
    fn=process_image,
    inputs="image",
    outputs="image",
)

demo.launch()
