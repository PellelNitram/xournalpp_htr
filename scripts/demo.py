import gradio as gr
from PIL import Image
import os

# --- Create a directory for downloads ---
if not os.path.exists("downloads"):
    os.makedirs("downloads")

# --- Image Processing Functions ---

def flip_image(image_path):
    """Flips the input image horizontally."""
    if image_path is None:
        return None
    # Open the image from the file path before processing
    with Image.open(image_path) as image:
        return image.transpose(Image.FLIP_LEFT_RIGHT)

def rotate_image(image_path):
    """Rotates the input image 90 degrees counter-clockwise."""
    if image_path is None:
        return None, None
    # Open the image from the file path before processing
    with Image.open(image_path) as image:
        rotated_img = image.rotate(90, expand=True)
        # Must return two values for the two outputs (viewer and state)
        return rotated_img, rotated_img

def save_for_download(image):
    """Saves the image to a file and returns the path."""
    if image is None:
        return None

    # Define a file path for the downloaded image
    download_path = "downloads/rotated_image.png"
    image.save(download_path)
    return download_path

# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Image Flipper and Rotator
        Upload a PNG file, then use the buttons below to manipulate it.
        """
    )

    # Hidden state components to store image data between button clicks
    original_image_state = gr.State()
    rotated_image_state = gr.State()

    # 1. Upload Button
    upload_button = gr.UploadButton(
        "Click to Upload a PNG File",
        file_types=["image"],
        file_count="single"
    )

    with gr.Row():
        # 2. Image Viewers
        image_viewer_1 = gr.Image(label="Flipped Image Viewer", interactive=False, height=350)
        image_viewer_2 = gr.Image(label="Rotated Image Viewer", interactive=False, height=350)

    with gr.Row():
        # 3. Action Buttons
        flip_btn = gr.Button("Show Flipped Image")
        rotate_btn = gr.Button("Show Rotated Image")

    # 4. Download Button and File Output
    download_btn = gr.Button("Download Rotated Image")
    download_file_output = gr.File(label="Download Link")


    # --- Event Handlers ---

    # When a user uploads a file, store its path in the 'original_image_state'
    # The output of UploadButton is a temporary file path
    upload_button.upload(
        lambda file: file,
        inputs=upload_button,
        outputs=original_image_state
    )

    # When the 'flip' button is clicked, process the original image and show it in viewer 1
    flip_btn.click(
        fn=flip_image,
        inputs=original_image_state,
        outputs=image_viewer_1
    )

    # When the 'rotate' button is clicked, process the original image, show it in viewer 2,
    # and also save the result to the 'rotated_image_state' for downloading later.
    rotate_btn.click(
        fn=rotate_image,
        inputs=original_image_state,
        outputs=[image_viewer_2, rotated_image_state]
    )

    # When the 'download' button is clicked, take the image from 'rotated_image_state',
    # save it to a file, and provide the file path to the download component.
    download_btn.click(
        fn=save_for_download,
        inputs=rotated_image_state,
        outputs=download_file_output
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use HF-provided port or fallback
    demo.launch(server_name="0.0.0.0", server_port=port)






# import os

# import gradio as gr

# from xournalpp_htr.documents import Stroke

# # TODO: Add gradio demo here.
# # - it needs to run locally
# # - it needs to run in a docker container that's compatible w/ huggingface space


# def greet(name):
#     s = Stroke  # Just for testing import
#     print(s)
#     return f"Hello, {name}!"


# with gr.Blocks() as demo:
#     gr.Markdown("# Greeting App")
#     with gr.Row():
#         name_input = gr.Textbox(
#             label="Enter your name", placeholder="Type your name here..."
#         )
#         output = gr.Textbox(label="Greeting")
#     greet_button = gr.Button("Greet")
#     greet_button.click(fn=greet, inputs=name_input, outputs=output)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 7860))  # Use HF-provided port or fallback
#     demo.launch(server_name="0.0.0.0", server_port=port)
