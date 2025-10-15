import os
import uuid
from pathlib import Path

import gradio as gr
from pdf2image import convert_from_path

from xournalpp_htr.documents import get_document
from xournalpp_htr.models import compute_predictions
from xournalpp_htr.utils import export_to_pdf_with_xournalpp
from xournalpp_htr.xio import write_predictions_to_PDF

# --- Image Processing Functions ---


def document_to_image_of_first_page(document_path):
    """Flips the input image horizontally."""
    if document_path is None:
        return None
    output_path = Path("/tmp/out.pdf")  # TODO: use path that is not hardcoded
    export_to_pdf_with_xournalpp(
        Path(document_path),
        output_path,
    )
    images = convert_from_path(output_path, first_page=1, last_page=1)
    first_page = images[0]
    return first_page


def document_to_HTR_document_and_image_of_first_page(document_path):
    """Rotates the input image 90 degrees counter-clockwise."""
    if document_path is None:
        return None, None
    document_path = Path(document_path)
    output_path = Path("/tmp/out.pdf")  # TODO: use path that is not hardcoded
    output_path_final = Path("/tmp/out_htr.pdf")  # TODO: use path that is not hardcoded
    document = get_document(document_path)
    predictions = compute_predictions(
        model_name="2024-07-18_htr_pipeline", document=document
    )
    write_predictions_to_PDF(
        output_path,
        output_path_final,
        predictions,
        debug_htr=True,
    )  # TODO: make it a generator to track progress externally like here.
    images = convert_from_path(output_path_final, first_page=1, last_page=1)
    first_page = images[0]
    return first_page, first_page


def save_HTR_document_for_download(session_id):
    path = Path("/tmp/out_htr.pdf")  # TODO: use path that is not hardcoded
    if not path.exists():
        return None
    return str(path)


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
        inputs=session_id,
        outputs=file_output,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use HF-provided port or fallback
    demo.launch(server_name="0.0.0.0", server_port=port)
