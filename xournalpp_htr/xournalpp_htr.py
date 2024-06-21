"""Script to perform HTR on Xournal(++) document."""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig

from documents import XournalDocument
from utils import export_to_pdf_with_xournalpp


def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-if', '--input-file', type=lambda p: Path(p).absolute(), required=True,
                        help='Path to the input Xournal or Xournal++ file.')
    parser.add_argument('-of', '--output-file', type=lambda p: Path(p).absolute(), required=True,
                        help='Path to the output PDF file.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='The model to use for handwriting recognition.') # TODO: Introduce dummy model called "test_lua_to_python"; TODO: Register models somehow to allow choice keyword here
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

    from pathlib import Path
    import subprocess
    import os
    import tempfile

    import pymupdf
    from pymupdf import TextWriter
    import cv2
    import matplotlib.pyplot as plt
    from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig
    from tqdm import tqdm

    # Goal
    #
    # Export as PDF: Write a script that uses XOJ as input and exports a PDF with text layer.

    # Settings

    XOJ_PATH = Path('../tests/test_1.xoj')
    PDF_PATH = XOJ_PATH.with_suffix('.pdf')
    PDF_PATH_OCR = PDF_PATH.parent / Path(PDF_PATH.stem+'_ocrd').with_suffix('.pdf')

    PREDICTION_IMAGE_DIR = None
    PREDICTION_IMAGE_DIR = Path('TEST_IMG') # Store prediction images in there for debugging purposes

    DEBUG_HTR = False # Debug switch. DEBUG_HTR=True leading to drawn boxes and visible text. Otherwise invisible text only.

    # Step 1: XOJ to PDF
    #
    # First, turn test file from `xoj` into `pdf`.

    export_to_pdf_with_xournalpp(args['input_file'], args['output_file'])

    # Step 2: Perform HTR predictions

    predictions = {}

    input_file = XOJ_PATH

    file_ending = input_file.suffix

    if file_ending == '.xoj':
        document = XournalDocument(input_file)
    else:
        raise NotImplementedError(f'File ending "{file_ending}" currently not readable.')

    nr_pages = len( document.pages )

    for page_index in tqdm(range(nr_pages)):

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

    if PREDICTION_IMAGE_DIR:

        PREDICTION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        nr_pages = len( document.pages )

        for page_index in tqdm(range(nr_pages)):

            file_name = PREDICTION_IMAGE_DIR / f'page{page_index}.jpg'
            file_name_ocrd = PREDICTION_IMAGE_DIR / f'page{page_index}_ocrd.jpg'

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

    try:
        os.remove(PDF_PATH_OCR)
    except Exception as e:
        print(f'Error encountered while deleting {PDF_PATH_OCR}', e)

    doc = pymupdf.open(PDF_PATH)

    input_file = XOJ_PATH

    nr_pages = len( document.pages )

    for page_index in tqdm(range(nr_pages)):

        pdf_page = doc[page_index]

        for prediction in predictions[page_index]:

            text = prediction['text']

            if DEBUG_HTR:

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
                render_mode=0 if DEBUG_HTR else 3, # 0 for visible, 3 for invisible
            ) # TODO: Improve text alignment with prediction. (1) center text vertically and then (2) stretch text to full box.
            #       Re (1) see https://github.com/pymupdf/PyMuPDF/discussions/1662.

    doc.ez_save(PDF_PATH_OCR)

    # Step 4: Next steps
    #
    # I want to build prediction code that can run both in a CLI and in a notebook like this here. Also, I'd like to be able to set the model flexibly.


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
