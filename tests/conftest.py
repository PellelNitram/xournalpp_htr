import urllib.request
from pathlib import Path

import pytest


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


@pytest.fixture
def get_path_to_IAM_OnDB_dataset(get_repo_root_directory: Path) -> Path:
    """TODO Add docstring."""

    path_to_IAM_OnDB_dataset = get_repo_root_directory / "data/datasets/IAM-OnDB"

    return path_to_IAM_OnDB_dataset
