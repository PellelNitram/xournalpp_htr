from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from xournalpp_htr.xio import load_IAM_OnDB_dataset


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
