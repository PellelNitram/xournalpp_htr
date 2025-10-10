import os
import gradio as gr
from pathlib import Path

# To make this example self-contained and runnable, we'll use reportlab
# to generate dummy PDF files.
# You'll need to install it: pip install reportlab
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except ImportError:
    print("Please install reportlab to run this demo: pip install reportlab")
    # Define a dummy class if reportlab is not installed so the app doesn't crash on import
    class canvas:
        def Canvas(self, *args, **kwargs): return self
        def drawString(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
    letter = (612, 792)


# --- Helper Function to Create Dummy PDFs ---
# This function creates a simple PDF file to simulate your app's output.
def create_dummy_pdf(file_path: Path, text: str):
    """Generates a simple PDF with the given text."""
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    c = canvas.Canvas(str(file_path), pagesize=letter)
    c.drawString(100, 750, text)
    c.save()
    print(f"Created dummy PDF: {file_path}")
    return str(file_path)


# --- Placeholder Functions for your Logic ---
# Replace the logic in these functions with your actual implementation.

def show_pdf_from_xoj(xoj_file):
    """
    Placeholder: This function should process the uploaded .xoj file
    and extract/generate the original PDF.
    """
    if xoj_file is None:
        return None, gr.update(visible=False), gr.update(visible=False), None

    print(f"Processing uploaded file: {xoj_file.name}")
    # For demo purposes, we create a dummy "original" PDF.
    # TODO: Replace this with your actual .xoj to PDF conversion logic.
    output_dir = Path("temp_output")
    original_pdf_path = create_dummy_pdf(
        output_dir / "original.pdf",
        "This is the original PDF extracted from the .xoj file."
    )
    # Make the HTR and Download buttons visible after a file is processed
    return original_pdf_path, gr.update(visible=True), gr.update(visible=True), None


def run_handwriting_recognition(original_pdf_path):
    """
    Placeholder: This function should take the original PDF, run HTR,
    and generate a new PDF with the recognized text.
    """
    if original_pdf_path is None:
        return None, None

    print(f"Running HTR on: {original_pdf_path}")
    # For demo purposes, we create a dummy "HTR result" PDF.
    # TODO: Replace this with your actual HTR processing logic.
    output_dir = Path("temp_output")
    htr_pdf_path = create_dummy_pdf(
        output_dir / "htr_result.pdf",
        "This is the new PDF with the HTR results."
    )
    return htr_pdf_path, htr_pdf_path


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Xournal++ Handwriting Recognition UI\n"
        "Upload a `.xoj` file, view the original PDF, and run HTR."
    )

    # 1. File Upload Section
    with gr.Row():
        upload_button = gr.UploadButton(
            "Click to Upload a .xoj File",
            # file_types=[".xoj"],
            file_count="single"
        )

    # 2. PDF Viewer Section
    with gr.Row(equal_height=True):
        # Use gr.File to display PDFs, as gr.PDF is deprecated.
        pdf_viewer_original = gr.File(label="Original PDF Viewer")
        pdf_viewer_htr = gr.File(label="HTR Result PDF Viewer")

    # 3. Action Buttons Section
    with gr.Row():
        htr_button = gr.Button("Run HTR", variant="primary", visible=False)

    # 4. Download Button Section
    with gr.Row():
        download_button = gr.DownloadButton("Download HTR PDF", visible=False)


    # --- Component Interactions ---

    # When a file is uploaded, process it, show the original PDF, and reset the HTR viewer.
    upload_button.upload(
        fn=show_pdf_from_xoj,
        inputs=upload_button,
        outputs=[pdf_viewer_original, htr_button, download_button, pdf_viewer_htr]
    )

    # When the HTR button is clicked, run the recognition and display the new PDF.
    # The output also populates the value of the download button.
    htr_button.click(
        fn=run_handwriting_recognition,
        inputs=pdf_viewer_original,
        outputs=[pdf_viewer_htr, download_button]
    )


if __name__ == "__main__":
    # Create a directory for dummy files if it doesn't exist
    if not os.path.exists("temp_output"):
        os.makedirs("temp_output")

    port = int(os.environ.get("PORT", 7860))
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
