from pathlib import Path

import numpy as np
import pytest
from src.data.online_handwriting_datasets import IAM_OnDB_Dataset

PATH = Path("data/datasets/IAM-OnDB")  # Needs to be parameterised


@pytest.mark.data
def test_construction_with_limit():
    limit = 5

    ds = IAM_OnDB_Dataset(path=PATH, transform=None, limit=limit)

    assert len(ds) == limit


@pytest.mark.data
@pytest.mark.slow
def test_construction_no_limit():
    ds = IAM_OnDB_Dataset(path=PATH, transform=None, limit=-1)

    assert len(ds) == IAM_OnDB_Dataset.LENGTH


@pytest.mark.data
@pytest.mark.slow
def test_construction_no_limit_skip_carbune2020_fails():
    ds = IAM_OnDB_Dataset(
        path=PATH, transform=None, limit=-1, skip_carbune2020_fails=True
    )

    assert len(ds) == IAM_OnDB_Dataset.LENGTH - len(
        IAM_OnDB_Dataset.SAMPLES_TO_SKIP_BC_CARBUNE2020_FAILS
    )


@pytest.mark.data
@pytest.mark.slow
def test_correctness_manually(tmp_path: Path):
    # This saves samples to files so that one can inspect the correctness of the
    # dataset manually. Enabling the pytest setting `-s` allows one to see where
    # the files were saved temporarily.

    NR_SAMPLES = 100
    LIMIT = -1

    print()
    print(f'Samples saved at: "{tmp_path}"')
    print()

    ds = IAM_OnDB_Dataset(path=PATH, transform=None, limit=LIMIT)

    # Get NR_SAMPLES reproducible random draws
    rng = np.random.default_rng(1337)
    index_list = np.arange(0, len(ds))
    rng.shuffle(index_list)
    index_list = index_list[:NR_SAMPLES]

    for iam_index in index_list:
        sample_name = ds[iam_index]["sample_name"]
        ds.plot_sample_to_image_file(iam_index, tmp_path / Path(f"{sample_name}.png"))
