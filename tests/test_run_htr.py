from pathlib import Path

import pytest

from xournalpp_htr.run_htr import main, parse_arguments


@pytest.mark.installation
def test_parse_arguments_empty() -> None:
    """Test `parse_arguments` with no command-line arguments.

    Ensures that `parse_arguments` raises a `SystemExit` exception when
    called without arguments. This typically occurs when required
    arguments are missing, triggering `sys.exit()`.

    Marked with `installation` for selective test runs.

    :returns: None
    """
    with pytest.raises(SystemExit):
        parse_arguments()


@pytest.mark.installation
def test_parse_arguments_full() -> None:
    """
    Test the `parse_arguments` function with a full set of input arguments.

    This test verifies that the `parse_arguments` function correctly
    parses a complete set of command-line arguments.
    """
    args = parse_arguments("-if input -of output -m dummy -pid dir -sp")
    assert len(args) == 5
    assert args["input_file"].stem == "input"
    assert args["output_file"].stem == "output"
    assert args["model"] == "dummy"
    assert args["prediction_image_dir"].stem == "dir"
    assert args["show_predictions"]


def test_main(get_path_to_minimal_test_data: Path, tmp_path: Path) -> None:
    """Tests the `main` function using minimal test data.

    Note: This function is currently not executed in Github Actions due to
    the requirement of a specific `Xournal++` version for the
    `export_to_pdf_with_xournalpp` function.

    :param get_path_to_minimal_test_data: Fixture to obtain path to the
                                          minimal test data file.
    :param tmp_path: Temporary path fixture used for storing the output
                     PDF file as a temporary file.
    """

    args = {
        "input_file": get_path_to_minimal_test_data,
        "output_file": tmp_path / Path("test_main.pdf"),
        "model": "2024-07-18_htr_pipeline",  # TODO: Add a `dummy` to test pipeline w/o ML part
        "prediction_image_dir": None,
        "show_predictions": False,
    }

    main(args)
