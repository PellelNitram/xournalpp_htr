import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from pdf2image import convert_from_path
from supabase import Client, create_client

from xournalpp_htr.documents import get_document
from xournalpp_htr.models import compute_predictions
from xournalpp_htr.utils import export_to_pdf_with_xournalpp, get_env_variable
from xournalpp_htr.xio import write_predictions_to_PDF

load_dotenv()

DEMO = get_env_variable("DEMO") == "1"
SB_URL = get_env_variable("SB_URL")
SB_KEY = get_env_variable("SB_KEY")
SB_BUCKET_NAME = get_env_variable("SB_BUCKET_NAME")
SB_SCHEMA_NAME = get_env_variable("SB_SCHEMA_NAME")
SB_TABLE_NAME = get_env_variable("SB_TABLE_NAME")

# --- Image Processing Functions ---


def get_temporary_directory() -> Path:
    return Path(tempfile.gettempdir())


def get_path_of_exported_pdf(session_id: str) -> Path:
    return get_temporary_directory() / f"{session_id}_input_as_pdf.pdf"


def get_path_of_pdf_with_htr(session_id: str) -> Path:
    return get_temporary_directory() / f"{session_id}_pdf_with_htr.pdf"


def log_interaction(
    session_id: str,
    donate_data: bool,
    interaction: str,
    document_path: str | None,
):
    supabase: Client = create_client(SB_URL, SB_KEY)

    if donate_data and document_path:
        document_path = Path(document_path)
        destination_path = f"{session_id}{document_path.suffix}"
        with open(document_path, "rb") as file:
            supabase.storage.from_(SB_BUCKET_NAME).upload(
                destination_path,
                file,
                {"content-type": "application/octet-stream"},
            )

    # Insert metadata row
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demo": DEMO,
        "session_id": session_id,
        "donate_data": donate_data,
        "interaction": interaction,
    }

    supabase.schema(SB_SCHEMA_NAME).table(SB_TABLE_NAME).insert(row).execute()


def upload_document(document_path, session_id: str, donate_data: bool) -> str:
    log_interaction(
        session_id=session_id,
        donate_data=donate_data,
        interaction="upload_document",
        document_path=document_path,
    )
    if document_path is None:
        return None
    return document_path


def document_to_image_of_first_page(document_path, session_id):
    """
    Converts the first page of the input document to an image.

    This function exports the input document to a PDF format using Xournal++,
    extracts the first page of the PDF, and converts it to an image. The image
    is then returned for further processing or display.

    Args:
        document_path (str or Path): The file path to the input document.
        session_id (str): The session identifier for logging and file management.

    Returns:
        PIL.Image.Image or None: The image of the first page of the exported PDF,
        or None if the input document path is not provided.

    Notes:
        - The function logs the interaction with the given session ID.
        - The exported PDF is stored in a temporary directory.
    """
    log_interaction(
        session_id=session_id,
        donate_data=False,
        interaction="document_to_image_of_first_page",
        document_path=None,
    )
    if document_path is None:
        return None
    output_path = get_path_of_exported_pdf(session_id)
    export_to_pdf_with_xournalpp(
        Path(document_path),
        output_path,
    )
    images = convert_from_path(output_path, first_page=1, last_page=1)
    first_page = images[0]
    return first_page


def document_to_HTR_document_and_image_of_first_page(document_path, session_id):
    """
    Processes a document to generate a PDF with handwritten text recognition (HTR) predictions
    and extracts the image of the first page.

    Args:
        document_path (str or Path): The file path to the input document.
        session_id (str): The session identifier for logging and file management.

    Returns:
        PIL.Image.Image or None: The image of the first page of the processed PDF, or None
        if the input document path is not provided.

    Notes:
        - The function logs the interaction with the given session ID.
        - Predictions are computed using a predefined HTR model.
        - The processed PDF with HTR predictions is generated and the first page is converted to an image.
        - The function currently assumes the use of a specific HTR model ("2024-07-18_htr_pipeline").
    """
    log_interaction(
        session_id=session_id,
        donate_data=False,
        interaction="document_to_HTR_document_and_image_of_first_page",
        document_path=None,
    )
    if document_path is None:
        return None
    document_path = Path(document_path)
    input_as_pdf_path = get_path_of_exported_pdf(session_id)
    pdf_with_htr = get_path_of_pdf_with_htr(session_id)
    document = get_document(document_path)
    predictions = compute_predictions(
        model_name="2024-07-18_htr_pipeline", document=document
    )
    write_predictions_to_PDF(
        input_as_pdf_path,
        pdf_with_htr,
        predictions,
        debug_htr=True,
    )  # TODO: make it a generator to track progress externally like here.
    images = convert_from_path(pdf_with_htr, first_page=1, last_page=1)
    first_page = images[0]
    return first_page


def save_HTR_document_for_download(session_id):
    """
    Saves the Handwritten Text Recognition (HTR) document for download.

    This function logs the interaction of saving an HTR document for download
    and retrieves the path of the PDF containing the HTR data. If the PDF does
    not exist, it returns None. Otherwise, it returns the string representation
    of the PDF's path.

    Args:
        session_id (str): The unique identifier for the user session.

    Returns:
        str or None: The file path of the PDF with HTR as a string if it exists,
        otherwise None.
    """
    log_interaction(
        session_id=session_id,
        donate_data=False,
        interaction="save_HTR_document_for_download",
        document_path=None,
    )
    pdf_with_htr = get_path_of_pdf_with_htr(session_id)
    if not pdf_with_htr.exists():
        return None
    return str(pdf_with_htr)


# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # [Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr) Demo

        This is an online demo of the [Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr) project, which strives to bring modern handwritten
        text recognition to open-source handwritten note softwares like [Xournal++](https://xournalpp.github.io/).

        While [Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr) is natively built to be running locally, this demo deploys it online so you
        can try it out without any installation. We do not collect any personal data (see [source code of this demo](https://github.com/PellelNitram/xournalpp_htr/blob/master/scripts/demo.py))
        but allow you to donate your data if you want so that we can build better underlying machine learning models for all of us (all open-source, of course!).

        Note that the HTR results are not yet perfect. This is an ongoing project and we are actively working on improving the models.
        Currently, we are constrained by the limited amount of publicly available training data and by our working time (this is a hobby project next to our day jobs).

        The "we" in the paragraphs above is currently really only me, [Martin Lellep](https://lellep.xyz/?utm_campaign=xppGradioDemo), the main developer of Xournal++ HTR. I really love to work on
        [Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr)! If you think this project is valuable and want to express your gratitute, then please feel free to [buy me a virtual coffee here](https://ko-fi.com/martin_l)
        so that I can buy more GPU power for training models and continue to let the GPUs go brrr :-).
        """
    )

    session_id = gr.State(
        value=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + str(uuid.uuid4())
    )

    original_image_state = gr.State()

    donate_data_checkbox = gr.Checkbox(
        label="Donate Data: Help us to improve our open-source models by donating your uploaded document. Everything will be released as open-source!",
        value=False,
    )

    upload_button = gr.UploadButton(
        "1. Click to Upload an XOJ File",
        file_types=[".xoj", ".xopp"],
        file_count="single",
    )

    with gr.Row():
        image_viewer_1 = gr.Image(
            label="Original document", interactive=False, height=350
        )
        image_viewer_2 = gr.Image(
            label="Document with HTR", interactive=False, height=350
        )

    with gr.Row():
        button_1 = gr.Button("2. Export to PDF and Show First Page")
        button_2 = gr.Button("3. Compute PDF with HTR and Show First Page")

    button_download = gr.Button("4. Download PDF with HTR")
    file_output = gr.File(label="Download PDF with HTR")

    # --- Event Handlers ---

    upload_button.upload(
        fn=upload_document,
        inputs=[upload_button, session_id, donate_data_checkbox],
        outputs=original_image_state,
    )

    button_1.click(
        fn=document_to_image_of_first_page,
        inputs=[original_image_state, session_id],
        outputs=image_viewer_1,
    )

    button_2.click(
        fn=document_to_HTR_document_and_image_of_first_page,
        inputs=[original_image_state, session_id],
        outputs=image_viewer_2,
    )

    button_download.click(
        fn=save_HTR_document_for_download,
        inputs=session_id,
        outputs=file_output,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use HF-provided port or fallback
    demo.launch(server_name="0.0.0.0", server_port=port)
