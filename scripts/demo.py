import os
import tempfile
import uuid

import gradio as gr
from PIL import Image

# --- Image Processing Functions ---


def document_to_image_of_first_page(document_path):
    """Flips the input image horizontally."""
    if document_path is None:
        return None
    # TODO:
    # 1. Use `xournalpp` to render file into PDF
    # 2. Load first page into image and return; check how to load into page
    with Image.open(document_path) as image:
        return image.transpose(Image.FLIP_LEFT_RIGHT)


def document_to_HTR_document_and_image_of_first_page(document_path):
    """Rotates the input image 90 degrees counter-clockwise."""
    if document_path is None:
        return None, None
    # TODO:
    # 1. Render document into HTR'd document
    #   - Either write `apply_htr_in_gradio_demo` function or use
    #     `export_xournalpp_to_pdf_with_htr`; probably the former
    # 2. Load first page of HTR'd document into image
    # 3. Return both image and full HTR'd document for display and
    #    download, respectively
    with Image.open(document_path) as image:
        rotated_img = image.rotate(90, expand=True)
        # Must return two values for the two outputs (viewer and state)
        return rotated_img, rotated_img


def save_HTR_document_for_download(image, session_id):
    """Saves the image to a file and returns the path."""
    if image is None:
        return None

    # Create a user-specific directory in temp
    user_downloads_dir = os.path.join(tempfile.gettempdir(), "downloads", session_id)
    os.makedirs(user_downloads_dir, exist_ok=True)

    # Create a unique filename to avoid conflicts
    unique_filename = f"rotated_image_{uuid.uuid4().hex[:8]}.png"
    download_path = os.path.join(user_downloads_dir, unique_filename)

    image.save(download_path)
    return download_path


# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Xournal++ HTR Demo

        This is an online demo of the [Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr) project, which strives to bring modern handwritten
        text recognition to open-source handwritten note softwares like [Xournal++](https://xournalpp.github.io/).

        While [Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr) is natively built to be running locally, this demo deploys it online for easier usage
        and the possibility to donate data for improving the underlying machine learning models.

        EXPLAIN this online demo in context of offline Xournal++ HTR abmitions, maybe w/ video?
        Upload a PNG file, then use the buttons below to manipulate it.
        """
    )

    session_id = gr.State(value=lambda: str(uuid.uuid4()))

    original_image_state = gr.State()
    rotated_image_state = gr.State()

    upload_button = gr.UploadButton(
        "Click to Upload an XOJ File", file_types=[".xoj"], file_count="single"
    )

    with gr.Row():
        image_viewer_1 = gr.Image(
            label="Original document", interactive=False, height=350
        )
        image_viewer_2 = gr.Image(label="HTR'd document", interactive=False, height=350)

    with gr.Row():
        button_1 = gr.Button("Show Flipped Image")
        button_2 = gr.Button("Show Rotated Image")

    button_download = gr.Button("Download Rotated Image")
    file_output = gr.File(label="Download Link")

    # --- Event Handlers ---

    upload_button.upload(
        lambda file: file, inputs=upload_button, outputs=original_image_state
    )

    button_1.click(
        fn=document_to_image_of_first_page,
        inputs=original_image_state,
        outputs=image_viewer_1,
    )

    button_2.click(
        fn=document_to_HTR_document_and_image_of_first_page,
        inputs=original_image_state,
        outputs=[image_viewer_2, rotated_image_state],
    )

    button_download.click(
        fn=save_HTR_document_for_download,
        inputs=[rotated_image_state, session_id],
        outputs=file_output,
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
