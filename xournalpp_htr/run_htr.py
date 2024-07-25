"""Script to perform HTR on Xournal(++) document."""

import argparse
from pathlib import Path

from xournalpp_htr.documents import get_document
from xournalpp_htr.utils import export_to_pdf_with_xournalpp
from xournalpp_htr.xio import get_temporary_filename
from xournalpp_htr.xio import write_predictions_to_PDF
from xournalpp_htr.models import compute_predictions
from xournalpp_htr.models import store_predictions_as_images


def parse_arguments(cli_string: None | str = None):
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-if', '--input-file', type=lambda p: Path(p).absolute(), required=True,
                        help='Path to the input Xournal or Xournal++ file.')
    parser.add_argument('-of', '--output-file', type=lambda p: Path(p).absolute(), required=True,
                        help='Path to the output PDF file.')
    # v-- TODO: Make optional
    parser.add_argument('-m', '--model', type=str, required=False, default='2024-07-18_htr_pipeline',
                        help='The model to use for handwriting recognition.') # TODO: Introduce dummy model called "test_lua_to_python"; TODO: Register models somehow to allow choice keyword here; also add "none"; default to latest model
    parser.add_argument('-pid', '--prediction-image-dir', type=lambda p: Path(p).absolute(), required=False,
                        help='If provided, images of the pages with overlaid '
                        'predictions are stored in the provided folder. '
                        'Useful for debugging purposes.')
    parser.add_argument('-sp', '--show-predictions', action='store_true',
                        help='Store the predictions and bounding boxes '
                        'visibly in the output file if enabled. '
                        'Useful for debugging purposes. '
                        'Otherwise only store invisible text.')
    args = vars( parser.parse_args(cli_string.split() if cli_string else None) )
    return args

def main(args: dict) -> None:
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

    input_file = args['input_file']
    prediction_image_dir = args['prediction_image_dir']
    output_file = args['output_file']
    debug_htr = args['show_predictions']
    model = args['model']

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

    print('xournalpp_htr: Done!')

    # Step 4: Next steps
    #
    # I want to build prediction code that can run both in a CLI and in a notebook like this here. Also, I'd like to be able to set the model flexibly.


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
