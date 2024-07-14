"""Script to perform HTR on Xournal(++) document."""

import argparse
from pathlib import Path
import os
import tempfile

import cv2
import matplotlib.pyplot as plt
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig
import pymupdf
from pymupdf import TextWriter
import cv2
import matplotlib.pyplot as plt
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig
from tqdm import tqdm

from documents import XournalDocument
from documents import XournalppDocument
from utils import export_to_pdf_with_xournalpp
from io import get_temporary_filename


def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-if', '--input-file', type=lambda p: Path(p).absolute(), required=True,
                        help='Path to the input Xournal or Xournal++ file.')
    parser.add_argument('-of', '--output-file', type=lambda p: Path(p).absolute(), required=True,
                        help='Path to the output PDF file.')
    # v-- TODO: Make optional
    parser.add_argument('-m', '--model', type=str, required=True,
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

    predictions = {}

    file_ending = input_file.suffix

    if file_ending == '.xoj':
        document = XournalDocument(input_file)
    elif file_ending == '.xopp':
        document = XournalppDocument(input_file)
    else:
        raise NotImplementedError(f'File ending "{file_ending}" currently not readable.')

    nr_pages = len( document.pages )

    for page_index in tqdm(range(nr_pages), desc='Recognition'):

        with tempfile.NamedTemporaryFile(dir='/tmp', delete=False, prefix=f'xournalpp_htr__page{page_index}__', suffix='.jpg') as tmpfile:
            TMP_FILE = Path(tmpfile.name)
        
            written_file = document.save_page_as_image(page_index, TMP_FILE, False, dpi=150)

            # ======
            # Do HTR
            # ======

            # read image
            img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)

            # detect and read text
            #height = 700 # good
            #enlarge = 5
            #enlarge = 10
            # height = 1000 # good
            # height = 1600 # not good
            scale = 0.4
            margin = 5
            read_lines = read_page(img, 
                                DetectorConfig(scale=scale, margin=margin), 
                                line_clustering_config=LineClusteringConfig(min_words_per_line=2))
            
            predictions_page = []
            for line in read_lines:
                for word in line:
                    data = {
                        'page_index': page_index,
                        'text': word.text,
                        'xmin': word.aabb.xmin,
                        'xmax': word.aabb.xmax,
                        'ymin': word.aabb.ymin,
                        'ymax': word.aabb.ymax,
                    }
                    predictions_page.append(data)
            predictions[page_index] = predictions_page

    # Plot the predictions to ensure that they are working correctly:

    if prediction_image_dir:

        prediction_image_dir.mkdir(parents=True, exist_ok=True)

        nr_pages = len( document.pages )

        for page_index in tqdm(range(nr_pages), desc='Store predictions'):

            file_name = prediction_image_dir / f'page{page_index}.jpg'
            file_name_ocrd = prediction_image_dir / f'page{page_index}_ocrd.jpg'

            written_file = document.save_page_as_image(page_index, file_name, False, dpi=150)

            # ======
            # Do HTR
            # ======

            # read image
            img = cv2.imread(str(written_file), cv2.IMREAD_GRAYSCALE)
        
            # To prepare plotting
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Impose predictions on image
            for prediction in predictions[page_index]:

                text = prediction['text']
                xmin = prediction['xmin']
                xmax = prediction['xmax']
                ymin = prediction['ymin']
                ymax = prediction['ymax']

                img = cv2.rectangle(img,
                                    (int(xmin), int(ymax)),
                                    (int(xmax), int(ymin)),
                                    (255, 0, 0),
                                    2)
                
                img = cv2.putText(img,
                                text=text,
                                org=(int(xmin), int(ymin)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(255, 0, 0),
                                thickness=1,
                                )
                    
            plt_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            figure_aspect_ratio = float(document.pages[page_index].meta_data['height']) / float(document.pages[page_index].meta_data['width'])
            plt.figure(figsize=(10, 10*figure_aspect_ratio))
            imgplot = plt.imshow(plt_image)
            plt.savefig(file_name_ocrd, dpi=150)
            plt.close()

    # Step 3: Store predictions in PDF

    doc = pymupdf.open(output_file_tmp_noOCR)

    nr_pages = len( document.pages )

    for page_index in tqdm(range(nr_pages), desc='Export to PDF'):

        pdf_page = doc[page_index]

        for prediction in predictions[page_index]:

            text = prediction['text']

            if debug_htr:

                pdf_page.draw_rect(
                    rect=pymupdf.Rect(
                        [
                            prediction['xmin'] / 150 * 72,
                            prediction['ymin'] / 150 * 72,
                        ],
                        [
                            prediction['xmax'] / 150 * 72,
                            prediction['ymax'] / 150 * 72,
                        ],
                    ),
                    color=pymupdf.pdfcolor["blue"],
                )

            pdf_page.insert_textbox(
                rect=pymupdf.Rect(
                    [
                        prediction['xmin'] / 150 * 72,
                        prediction['ymin'] / 150 * 72,
                    ],
                    [
                        prediction['xmax'] / 150 * 72,
                        prediction['ymax'] / 150 * 72,
                    ],
                ),
                buffer=prediction['text'],
                color=pymupdf.pdfcolor["blue"],
                align=pymupdf.TEXT_ALIGN_CENTER,
                fontsize=6,
                render_mode=0 if debug_htr else 3, # 0 for visible, 3 for invisible
            ) # TODO: Improve text alignment with prediction. (1) center text vertically and then (2) stretch text to full box.
            #       Re (1) see https://github.com/pymupdf/PyMuPDF/discussions/1662.

    doc.ez_save(output_file)

    # Step 4: Next steps
    #
    # I want to build prediction code that can run both in a CLI and in a notebook like this here. Also, I'd like to be able to set the model flexibly.


if __name__ == '__main__':
    args = parse_arguments()
    main(args)