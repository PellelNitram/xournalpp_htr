import urllib.request
from pathlib import Path

import pytest

from xournalpp_htr.run_htr import main, parse_arguments


@pytest.fixture
def get_repo_root_directory(request):
    """TODO.

    Fixture based on [this](https://stackoverflow.com/a/57039134).
    """
    rootdir = Path(request.config.rootdir)
    assert (rootdir / "README.md").is_file()
    return rootdir


@pytest.fixture
def get_path_to_minimal_test_data(get_repo_root_directory):
    """TODO. Get path to minimal test data file.

    If the file does not exist, then it is created at `${repo_root}/tests/data/2024-07-26_minimal.xopp`
    """

    path_to_minimal_test_data = (
        get_repo_root_directory / "tests/data/2024-07-26_minimal.xopp"
    )

    if not path_to_minimal_test_data.is_file():
        url = "https://bit.ly/2024-07-26_minimal_xopp"
        urllib.request.urlretrieve(url, path_to_minimal_test_data)

    return path_to_minimal_test_data


@pytest.mark.installation
def test_parse_arguments_empty():
    with pytest.raises(SystemExit):
        parse_arguments()


@pytest.mark.installation
def test_parse_arguments_full():
    args = parse_arguments("-if input -of output -m dummy -pid dir -sp")
    assert len(args) == 5
    assert args["input_file"].stem == "input"
    assert args["output_file"].stem == "output"
    assert args["model"] == "dummy"
    assert args["prediction_image_dir"].stem == "dir"
    assert args["show_predictions"]


def test_main(get_path_to_minimal_test_data, tmp_path):
    """TODO.

    This is not checked in Github Actions for now b/c I would have to install
    the right `Xournal++` version there to allow `export_to_pdf_with_xournalpp`
    to work.
    """

    args = {
        "input_file": get_path_to_minimal_test_data,
        "output_file": tmp_path / Path("test_main.pdf"),
        "model": "2024-07-18_htr_pipeline",  # TODO: Add a `dummy` to test pipeline w/o ML part
        "prediction_image_dir": None,
        "show_predictions": False,
    }

    main(args)
