# TODO: Rename to `io` once `xournalpp_htr.py` was moved from this folder.

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from xournalpp_htr.models import PageIndex, WordPrediction

try:
    from huggingface_hub import snapshot_download

    huggingface_hub_available = True
except ImportError:
    huggingface_hub_available = False
from tqdm import tqdm


@dataclass
class BenchmarkSample:
    xopp_path: Path
    gt_path: Path


def write_predictions_to_PDF(
    input_pdf_file: Path,
    output_pdf_file: Path,
    predictions: dict[PageIndex, list[WordPrediction]],
    debug_htr: bool,
    small_text: bool = False,
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

    # Cache font metrics for Helvetica (the default font used by insert_text).
    _helv = pymupdf.Font("helv")
    # Scale factor: fills ~95% of the bounding box height with a single text line.
    _fontsize_fill_factor = 0.95 / (_helv.ascender - _helv.descender)
    # In PyMuPDF (y-down), char_center_y = baseline_y - ascender*fs + line_h/2
    # Rearranging: baseline_y = char_center_y + (ascender + descender)/2 * fs
    _char_center_to_baseline = (_helv.ascender + _helv.descender) / 2

    for page_index in tqdm(range(nr_pages), desc="Export to PDF"):
        pdf_page = doc[page_index]

        for prediction in predictions[page_index]:
            rect = pymupdf.Rect(
                [prediction.xmin, prediction.ymin],
                [prediction.xmax, prediction.ymax],
            )

            if debug_htr:
                pdf_page.draw_rect(rect=rect, color=pymupdf.pdfcolor["blue"])

            # Scale the fontsize to fill the prediction bounding box height so
            # that text selection in PDF viewers aligns with the handwriting.
            # Place the baseline so characters are vertically centred on the
            # box, and horizontally centre the word within it.
            box_h = prediction.ymax - prediction.ymin
            box_w = prediction.xmax - prediction.xmin
            if small_text:
                fontsize = 6
            else:
                fontsize_from_height = box_h * _fontsize_fill_factor
                fontsize_from_width = box_w / _helv.text_length(prediction.text, 1.0)
                fontsize = min(fontsize_from_height, fontsize_from_width)

            box_center_y = (prediction.ymin + prediction.ymax) / 2
            baseline_y = box_center_y + _char_center_to_baseline * fontsize

            text_width = _helv.text_length(prediction.text, fontsize)
            x_start = (prediction.xmin + prediction.xmax) / 2 - text_width / 2

            pdf_page.insert_text(
                point=(x_start, baseline_y),
                text=prediction.text,
                fontsize=fontsize,
                color=pymupdf.pdfcolor["blue"],
                render_mode=0 if debug_htr else 3,
            )

    doc.ez_save(output_pdf_file)


def get_temporary_filename() -> Path:
    """
    Generates and returns a temporary PDF file name.

    This function creates a named temporary file in `/tmp` using `tempfile.NamedTemporaryFile` with a `xournalpp_htr`
    specific prefix and PDF suffix. The generated filename is returned as a `Path` object. To ensure that this method
    also works on Windows, the parent folder of the temporary file is created just to be on the safe side.

    :return: A `Path` object representing the temporary PDF file name.
    """

    with tempfile.NamedTemporaryFile(
        dir="/tmp", delete=True, prefix="xournalpp_htr__tmp_pdf_export__", suffix=".pdf"
    ) as tmp_file_manager:
        output_file_tmp_noOCR = Path(tmp_file_manager.name)

    output_file_tmp_noOCR.parent.mkdir(parents=True, exist_ok=True)

    return output_file_tmp_noOCR


def load_benchmark() -> list[BenchmarkSample]:
    """Return benchmark samples from the xournalpp_htr_benchmark HuggingFace dataset."""
    if not huggingface_hub_available:
        raise ImportError(
            "The `huggingface_hub` package is required to load the benchmark data."
        )
    local_dir = Path(
        snapshot_download("PellelNitram/xournalpp_htr_benchmark", repo_type="dataset")
    )
    data_dir = local_dir / "data"
    xopp_files = sorted(data_dir.glob("*.xopp")) + sorted(data_dir.glob("*.xoj"))
    samples = []
    for xopp_path in xopp_files:
        gt_path = xopp_path.with_suffix("").with_suffix(".gt.json")
        if gt_path.exists():
            samples.append(BenchmarkSample(xopp_path=xopp_path, gt_path=gt_path))
    return samples


def load_IAM_OnDB_dataset() -> Path:
    """Return path to the IAM-OnDB dataset, downloading from HuggingFace Hub if needed.

    Requires a valid HuggingFace token with access to the private repository,
    set via the ``HF_TOKEN`` environment variable or ``huggingface-cli login``.

    :returns: Path to the root of the IAM-OnDB dataset (the ``data/`` subfolder
              of the HuggingFace repo).
    """
    if not huggingface_hub_available:
        raise ImportError(
            "The `huggingface_hub` package is required to load the IAM-OnDB dataset."
        )
    return (
        Path(
            snapshot_download(
                repo_id="PellelNitram/xournalpp_htr_IAM_OnDB",
                repo_type="dataset",
            )
        )
        / "data"
    )


def load_examples(exclude_empty: bool = False):
    if not huggingface_hub_available:
        raise ImportError(
            "The `huggingface_hub` package is required to load the example data."
        )
    repo_id = "PellelNitram/xournalpp_htr_examples"

    extensions = {".xoj", ".xopp"}

    # Download the repo locally
    local_dir = snapshot_download(repo_id, repo_type="dataset")

    # Point to the data folder
    data_dir = Path(local_dir) / "data"

    # Collect all matching file paths recursively
    file_paths = sorted(
        [str(f) for f in data_dir.rglob("*") if f.suffix.lower() in extensions]
    )

    # Remove empty files if requested
    if exclude_empty:
        file_paths = [fp for fp in file_paths if "empty" not in Path(fp).stem.lower()]

    return file_paths
