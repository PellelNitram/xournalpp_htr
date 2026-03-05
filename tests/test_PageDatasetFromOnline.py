from pathlib import Path

import pytest
import torch

from xournalpp_htr.training.data.datasets import (
    IAM_OnDB_Dataset,
    PageDatasetFromOnline,
    PageDatasetFromOnlinePosition,
)


def test_compute(get_path_to_IAM_OnDB_dataset: Path, tmp_path: Path):
    limit = 100
    ds = IAM_OnDB_Dataset(
        path=get_path_to_IAM_OnDB_dataset, transform=None, limit=limit
    )

    positions_height = 15.0
    positions = [
        PageDatasetFromOnlinePosition(
            stroke_width=1,
            page_index=0,
            center_x=105.0,
            center_y=100.0,
            height=positions_height,
            dataset_index=0,
        ),
        PageDatasetFromOnlinePosition(
            stroke_width=1,
            page_index=0,
            center_x=105.0,
            center_y=200.0,
            height=positions_height,
            dataset_index=1,
        ),
    ]

    print(len(ds.data))
    p_ds = PageDatasetFromOnline(
        dataset=ds,
        positions=positions,
        page_size=(297, 210),
        cache_dir=tmp_path,
        dpi=72,
    )
    result = p_ds.compute()

    assert len(result) > 0


@pytest.mark.visual_check
def test_render_page_and_mask(get_path_to_IAM_OnDB_dataset: Path, tmp_path: Path):
    print(f"\n\nPath to check out: {tmp_path}.\n")

    limit = 100
    ds = IAM_OnDB_Dataset(
        path=get_path_to_IAM_OnDB_dataset, transform=None, limit=limit
    )

    positions_height = 15.0
    positions = [
        PageDatasetFromOnlinePosition(
            stroke_width=1,
            page_index=0,
            center_x=105.0,
            center_y=100.0,
            height=positions_height,
            dataset_index=0,
        ),
        PageDatasetFromOnlinePosition(
            stroke_width=1,
            page_index=0,
            center_x=105.0,
            center_y=200.0,
            height=positions_height,
            dataset_index=1,
        ),
    ]

    p_ds = PageDatasetFromOnline(
        dataset=ds,
        positions=positions,
        page_size=(297, 210),
        cache_dir=tmp_path,
        dpi=72,
    )
    for key in p_ds.data:
        p_ds.render_page_and_mask(
            page_index=key,
            output_path_page=tmp_path / f"test_page_{key}.png",
            output_path_mask=tmp_path / f"test_mask_{key}.png",
        )


@pytest.mark.visual_check
def test_getitem(get_path_to_IAM_OnDB_dataset: Path, tmp_path: Path):
    print(f"\n\nPath to check out: {tmp_path}.\n")

    limit = 100
    ds = IAM_OnDB_Dataset(
        path=get_path_to_IAM_OnDB_dataset, transform=None, limit=limit
    )

    positions_height = 15.0
    positions = [
        PageDatasetFromOnlinePosition(
            stroke_width=1,
            page_index=0,
            center_x=105.0,
            center_y=100.0,
            height=positions_height,
            dataset_index=0,
        ),
        PageDatasetFromOnlinePosition(
            stroke_width=1,
            page_index=0,
            center_x=105.0,
            center_y=200.0,
            height=positions_height,
            dataset_index=1,
        ),
    ]

    p_ds = PageDatasetFromOnline(
        dataset=ds,
        positions=positions,
        page_size=(297, 210),
        cache_dir=tmp_path,
        dpi=72,
    )
    sample = p_ds[0]

    image = sample["image"]
    segmentation_mask = sample["segmentation_mask"]

    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3
    assert image.dtype == torch.uint8
    assert image.shape[1:] == segmentation_mask.shape[1:]
    assert segmentation_mask.shape[0] == 3
    assert segmentation_mask.dtype == torch.uint8


# TODO: Add tests for all `PageDatasetFromOnline` functions - really?
