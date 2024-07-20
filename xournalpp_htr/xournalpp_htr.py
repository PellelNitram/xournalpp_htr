"""Script to perform HTR on Xournal(++) document."""

import argparse
from pathlib import Path
import os

import cv2
import matplotlib.pyplot as plt
import pymupdf
from pymupdf import TextWriter
import cv2
import matplotlib.pyplot as plt
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig
from tqdm import tqdm

from documents import XournalDocument
from documents import XournalppDocument
from documents import get_document
from utils import export_to_pdf_with_xournalpp
from xio import get_temporary_filename
from xio import write_predictions_to_PDF
from models import compute_predictions
from models import store_predictions_as_images


def parse_arguments():
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
    args = vars( parser.parse_args() )
    return args

def main(args):
    """TODO.
    
    TODO: This is what I do here:
    1. export X file to PDF
    2. do HTR on X file using model that is specified by `parse_arguments`
    3. store prediction as hidden text to PDF file

    # TODO: Define coordinate transforms between X file, prediction and PDF as matrices like in computer graphics.
    """

    # Goal
    #
    # Export as PDF: Write a script that uses XOJ as input and exports a PDF with text layer.

    # Settings

    input_file = args['input_file']
    prediction_image_dir = args['prediction_image_dir']
    output_file = args['output_file']
    debug_htr = args['show_predictions']

    output_file_tmp_noOCR = get_temporary_filename()

    # Step 1: XOJ to PDF
    #
    # First, turn test file from `xoj` into `pdf`.

    export_to_pdf_with_xournalpp(input_file, output_file_tmp_noOCR)

    # Step 2: Perform HTR predictions

    document = get_document(input_file)

    predictions = compute_predictions(model_name=args['model'], document=document)

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
