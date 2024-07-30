# TODO: Rename to `io` once `xournalpp_htr.py` was moved from this folder.

import tempfile
from pathlib import Path

import pymupdf
from tqdm import tqdm


def write_predictions_to_PDF(
    input_pdf_file: Path,
    output_pdf_file: Path,
    predictions: dict,
    debug_htr: bool,
) -> None:
    """
    Writes handwritten text predictions to a PDF file.

    This function reads an input PDF file using PyMuPDF, extracts each page from the PDF, and then adds predictions to each page in the form of rectangles and text boxes.
    The rectangles are drawn if `debug_htr` is True, otherwise they are not. The text boxes contain the predicted text and are visible if `debug_htr` is True, otherwise
    they are invisible.

    :param input_pdf_file: The input PDF file.
    :param output_pdf_file: The output PDF file.
    :param predictions: A dictionary of page indices to lists of predictions, where each prediction is a dictionary containing `text`, `xmin`, `ymin`, `xmax`, and `ymax` keys.
    :param debug_htr: Whether to draw rectangles around the predicted regions and render visible text boxes. If False, the rectangles are not drawn and invisible text boxes are rendered.
    :returns: Nothing.
    """

    doc = pymupdf.open(input_pdf_file)

    nr_pages = len(doc)

    for page_index in tqdm(range(nr_pages), desc="Export to PDF"):
        pdf_page = doc[page_index]

        for prediction in predictions[page_index]:
            if debug_htr:
                pdf_page.draw_rect(
                    rect=pymupdf.Rect(
                        [
                            prediction["xmin"] / 150 * 72,
                            prediction["ymin"] / 150 * 72,
                        ],
                        [
                            prediction["xmax"] / 150 * 72,
                            prediction["ymax"] / 150 * 72,
                        ],
                    ),
                    color=pymupdf.pdfcolor["blue"],
                )

            pdf_page.insert_textbox(
                rect=pymupdf.Rect(
                    [
                        prediction["xmin"] / 150 * 72,
                        prediction["ymin"] / 150 * 72,
                    ],
                    [
                        prediction["xmax"] / 150 * 72,
                        prediction["ymax"] / 150 * 72,
                    ],
                ),
                buffer=prediction["text"],
                color=pymupdf.pdfcolor["blue"],
                align=pymupdf.TEXT_ALIGN_CENTER,
                fontsize=6,
                render_mode=0 if debug_htr else 3,  # 0 for visible, 3 for invisible
            )
            # TODO: Improve text alignment with prediction. (1) center text vertically and then (2) stretch text to full box.
            #       Re (1) see https://github.com/pymupdf/PyMuPDF/discussions/1662.

    doc.ez_save(output_pdf_file)


def get_temporary_filename() -> Path:
    """
    Generates and returns a temporary PDF file name.

    This function creates a named temporary file in `/tmp` using `tempfile.NamedTemporaryFile` with a `xournalpp_htr`
    specific prefix and PDF suffix. The generated filename is returned as a `Path` object.

    :return: A `Path` object representing the temporary PDF file name.
    """

    with tempfile.NamedTemporaryFile(
        dir="/tmp", delete=True, prefix="xournalpp_htr__tmp_pdf_export__", suffix=".pdf"
    ) as tmp_file_manager:
        output_file_tmp_noOCR = Path(tmp_file_manager.name)

    return output_file_tmp_noOCR
