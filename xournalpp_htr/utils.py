import argparse
import os
import subprocess
from pathlib import Path


def export_to_pdf_with_xournalpp(input_file: Path, output_file: Path) -> None:
    """Export a Xournal(++) file to PDF using Xournal++.

    This function uses the `xournalpp` command-line tool to convert a Xournal(++) file
    specified by `input_file` to a PDF file specified by `output_file`. If the export fails
    for any reason, an exception is raised.

    :param input_file: Path to the input Xournal(++) file.
    :type input_file: Path
    :param output_file: Path to the output PDF file.
    :type output_file: Path
    :raises RuntimeError: If the PDF export fails.

    .. note::
       Ensure that Xournal++ is installed and available in the system's PATH.

    .. code-block:: python

        from pathlib import Path
        from xournalpp_htr.utils import export_to_pdf_with_xournalpp

        input_path = Path('/path/to/input/file.xopp')
        output_path = Path('/path/to/output/file.pdf')

        try:
            export_to_pdf_with_xournalpp(input_path, output_path)
            print("Export successful!")
        except RuntimeError as e:
            print(f"Export failed: {e}")
    """

    if not input_file.exists():
        raise ValueError(f'input file "{input_file}" does not exist.')

    command = f'xournalpp "{input_file}" -p "{output_file}"'
    export_result = subprocess.run(command, shell=True, capture_output=True)

    return_code_fail = export_result.returncode != 0
    stdout_fail = "PDF file successfully created" not in export_result.stderr.decode(
        "utf-8"
    )
    file_existing_fail = not output_file.exists()

    if return_code_fail or stdout_fail or file_existing_fail:
        raise RuntimeError(
            f"PDF export failed: {return_code_fail=}, {stdout_fail=}, {file_existing_fail=}"
        )

    return output_file


def parse_arguments(cli_string: None | str = None):
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-if",
        "--input-file",
        type=lambda p: Path(p).absolute(),
        required=True,
        help="Path to the input Xournal or Xournal++ file.",
    )
    parser.add_argument(
        "-of",
        "--output-file",
        type=lambda p: Path(p).absolute(),
        required=True,
        help="Path to the output PDF file.",
    )
    # v-- TODO: Make optional
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default="2024-07-18_htr_pipeline",
        help="The model to use for handwriting recognition.",
    )  # TODO: Introduce dummy model called "test_lua_to_python"; TODO: Register models somehow to allow choice keyword here; also add "none"; default to latest model
    parser.add_argument(
        "-pid",
        "--prediction-image-dir",
        type=lambda p: Path(p).absolute(),
        required=False,
        help="If provided, images of the pages with overlaid "
        "predictions are stored in the provided folder. "
        "Useful for debugging purposes.",
    )
    parser.add_argument(
        "-sp",
        "--show-predictions",
        action="store_true",
        help="Store the predictions and bounding boxes "
        "visibly in the output file if enabled. "
        "Useful for debugging purposes. "
        "Otherwise only store invisible text.",
    )
    args = vars(parser.parse_args(cli_string.split() if cli_string else None))
    return args


def get_env_variable(name: str, default=None):
    """
    Retrieve the value of an environment variable.

    Args:
        name (str): The name of the environment variable to retrieve.
        default (optional): The default value to return if the environment
            variable is not set. Defaults to None.

    Returns:
        The value of the environment variable if it is set, or the default
        value if provided.

    Raises:
        ValueError: If the environment variable is not set and no default
        value is provided.
    """
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable '{name}' is not set.")
    return value
