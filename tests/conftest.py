from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from xournalpp_htr.xio import load_IAM_OnDB_dataset


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
def get_path_to_minimal_test_data() -> Path:
    """Fixture to retrieve the path to the minimal test data file.

    Downloads and caches the file from the HuggingFace Hub dataset
    ``PellelNitram/xournalpp_htr_examples``.

    :returns: The path to the minimal test data file.
    """
    return Path(
        hf_hub_download(
            repo_id="PellelNitram/xournalpp_htr_examples",
            filename="data/2024-07-26_minimal.xopp",
            repo_type="dataset",
        )
    )


@pytest.fixture
def get_path_to_IAM_OnDB_dataset() -> Path:
    """Fixture to retrieve the path to the IAM-OnDB dataset.

    Downloads and caches the dataset from HuggingFace Hub if not already
    present locally. Requires a valid HuggingFace token with access to the
    private repository, set via the ``HF_TOKEN`` environment variable or
    ``huggingface-cli login``.

    :returns: Path to the root of the IAM-OnDB dataset (the ``data/``
              subfolder of the HuggingFace repo).
    """
    return load_IAM_OnDB_dataset()
