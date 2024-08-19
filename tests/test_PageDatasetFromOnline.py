from pathlib import Path

import matplotlib.pyplot as plt

from xournalpp_htr.training.data.datasets import (
    IAM_OnDB_Dataset,
    PageDatasetFromOnline,
    PageDatasetFromOnlinePosition,
)


def test_compute(get_path_to_IAM_OnDB_dataset: Path):
    limit = 100
    ds = IAM_OnDB_Dataset(
        path=get_path_to_IAM_OnDB_dataset, transform=None, limit=limit
    )

    positions_height = 5.0
    positions = [
        PageDatasetFromOnlinePosition(
            stroke_width=1,
            page_index=0,
            center_x=10.0,
            center_y=10.0,
            height=positions_height,
            dataset_index=0,
        ),
        PageDatasetFromOnlinePosition(
            stroke_width=1,
            page_index=0,
            center_x=10.0,
            center_y=20.0,
            height=positions_height,
            dataset_index=1,
        ),
    ]

    print(len(ds.data))
    p_ds = PageDatasetFromOnline(
        dataset=ds,
        positions=positions,
        page_size=(10, 10),
    )
    result = p_ds.compute()
    p_ds.render_pages()

    for page_index in result:
        plt.figure(figsize=(p_ds.page_size))
        for data in result[page_index]:
            for stroke_nr in data["strokes"]:
                x = data["strokes"][stroke_nr]["x"]
                y = data["strokes"][stroke_nr]["y"]
                # label = data["label"]
                plt.plot(
                    x,
                    y,
                    c="black",
                )
        plt.gca().set_aspect("equal")
        plt.show()
        plt.close()

    # TODO: construct segmentation mask as well


def test_render_pages():
    raise NotImplementedError("TODO")
    pass
    # TODO: Make sure to check the pages visually
