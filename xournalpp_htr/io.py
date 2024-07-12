from pathlib import Path

import pymupdf
import tqdm


def write_predictions_to_PDF(
    input_pdf_file: Path,
    output_pdf_file: Path,
    predictions: dict,
    debug_htr: bool,
):
    """TODO!"""

    doc = pymupdf.open(input_pdf_file)

    nr_pages = len( doc )

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

    doc.ez_save(output_pdf_file)