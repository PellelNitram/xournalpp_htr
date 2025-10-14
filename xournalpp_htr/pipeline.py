from xournalpp_htr.documents import get_document
from xournalpp_htr.models import compute_predictions, store_predictions_as_images
from xournalpp_htr.utils import export_to_pdf_with_xournalpp
from xournalpp_htr.xio import get_temporary_filename, write_predictions_to_PDF


def export_xournalpp_to_pdf_with_htr(args: dict) -> None:
    """Main function that performs HTR.

    This function exports an Xournal(++) file to a PDF file, performs Handwritten Text Recognition (HTR) on the file using
    the specified model, and stores the predictions as hidden text in the resulting PDF file. The function can also plot
    the predictions to ensure they are working correctly and save them to a directory for later inspection.

    This function can be imported elsewhere to use, e.g., in Jupyter notebooks or tests.

    :param args: Dictionary containing the following input parameters: `input_file' (Path; path to the input Xournal(++) file),
                 'prediction_image_dir' (Path or None; directory for storing prediction images (optional)), 'output_file'
                 (Path; path to the output PDF file), 'model' (str; name of the HTR model to use for predictions) and
                 `show_predictions` (bool; switch to render visible prediction texts in PDF instead of invisible texts).
    :returns: None
    """

    # Goal
    #
    # Export as PDF: Write a script that uses XOJ as input and exports a PDF with text layer.

    # Settings

    input_file = args["input_file"]
    prediction_image_dir = args["prediction_image_dir"]
    output_file = args["output_file"]
    debug_htr = args["show_predictions"]
    model = args["model"]

    output_file_tmp_noOCR = get_temporary_filename()

    # Step 1: XOJ to PDF
    #
    # First, turn test file from `xoj` into `pdf`.

    export_to_pdf_with_xournalpp(input_file, output_file_tmp_noOCR)

    # Step 2: Perform HTR predictions

    document = get_document(input_file)

    predictions = compute_predictions(model_name=model, document=document)

    # Plot the predictions to ensure that they are working correctly:

    if prediction_image_dir:
        store_predictions_as_images(prediction_image_dir, predictions, document)

    # Step 3: Store predictions in PDF
    write_predictions_to_PDF(
        output_file_tmp_noOCR,
        output_file,
        predictions,
        debug_htr,
    )

    print("xournalpp_htr: Done!")
