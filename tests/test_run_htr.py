import urllib.request
from pathlib import Path

import pytest

from xournalpp_htr.pipeline import export_xournalpp_to_pdf_with_htr
from xournalpp_htr.run_htr import parse_arguments


@pytest.fixture
def get_repo_root_directory(request: pytest.FixtureRequest) -> Path:
    """
    A `pytest` fixture that returns the root directory of the repository.

    This fixture retrieves the root directory from the pytest configuration
    and asserts that the directory contains a "README.md" file. It is based
    on an example provided in a
    [Stack Overflow answer](https://stackoverflow.com/a/57039134).

    :param request: A `pytest` fixture that provides information about the
                    requesting test function.
    :returns: A Path object representing the root directory of the repository.
    :raises AssertionError: If the "README.md" file is not found in the root
                            directory.
    """
    rootdir = Path(request.config.rootdir)
    assert (rootdir / "README.md").is_file()
    return rootdir


@pytest.fixture
def get_path_to_minimal_test_data(get_repo_root_directory: Path) -> Path:
    """Fixture to retrieve the path to the minimal test data file.

    This function checks for the existence of a specific test data file
    located at `${repo_root}/tests/data/2024-07-26_minimal.xopp`. If the
    file does not exist, it is downloaded from a predefined URL and saved
    at the specified location so that it can be retrieved directly next
    time; i.e. it is cached.

    :param get_repo_root_directory: A fixture that provides the root
                                    directory of the repository.
    :returns: The path to the minimal test data file.
    """

    path_to_minimal_test_data = (
        get_repo_root_directory / "tests/data/2024-07-26_minimal.xopp"
    )

    if not path_to_minimal_test_data.is_file():
        url = "https://bit.ly/2024-07-26_minimal_xopp"
        urllib.request.urlretrieve(url, path_to_minimal_test_data)

    return path_to_minimal_test_data


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

    export_xournalpp_to_pdf_with_htr(args)
