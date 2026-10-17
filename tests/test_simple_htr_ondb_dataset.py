import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from xournalpp_htr.training.simple_htr_ondb.dataset import (  # noqa: E402
    OnDB_Words_Dataset,
    custom_collate_fn,
)


def _write_manifest(tmp_path, rows):
    (tmp_path / "images").mkdir()
    with open(tmp_path / "manifest.jsonl", "w") as f:
        for name, text, split in rows:
            rel = f"images/{name}.png"
            cv2.imwrite(str(tmp_path / rel), np.full((40, 90), 200, dtype=np.uint8))
            f.write(json.dumps({"image": rel, "text": text, "split": split}) + "\n")


def test_dataset_reads_only_its_split_and_shares_the_train_charset(tmp_path):
    _write_manifest(
        tmp_path, [("a", "ab", "train"), ("b", "ba", "train"), ("c", "az", "val")]
    )

    train = OnDB_Words_Dataset(tmp_path, "train", 32, 128)
    val = OnDB_Words_Dataset(tmp_path, "val", 32, 128, charset=train.charset)

    assert len(train) == 2 and len(val) == 1
    assert train.charset == ["a", "b"]
    # "z" is not in the train charset: unencodable as a target, text stays raw.
    assert val[0]["encoded"] == [0]
    assert val[0]["text"] == "az"
    assert train[0]["image"].shape == (32, 128)


def test_collate_flattens_targets(tmp_path):
    _write_manifest(tmp_path, [("a", "ab", "train"), ("b", "b", "train")])
    train = OnDB_Words_Dataset(tmp_path, "train", 32, 128)

    batch = custom_collate_fn([train[0], train[1]])

    assert batch["images"].shape == (2, 1, 32, 128)
    assert batch["target_lengths"].tolist() == [2, 1]
    assert batch["targets"].tolist() == [0, 1, 1]
