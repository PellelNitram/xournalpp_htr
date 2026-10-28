import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from xournalpp_htr.training.simple_htr_ondb.dataset import (  # noqa: E402
    Words_Dataset,
    custom_collate_fn,
    load_iam_samples,
    load_ondb_samples,
)


def _write_manifest(tmp_path, rows):
    (tmp_path / "images").mkdir()
    with open(tmp_path / "manifest.jsonl", "w") as f:
        for name, text, split in rows:
            rel = f"images/{name}.png"
            cv2.imwrite(str(tmp_path / rel), np.full((40, 90), 200, dtype=np.uint8))
            f.write(json.dumps({"image": rel, "text": text, "split": split}) + "\n")


def _write_iam(root, words):
    """words: list of (word_id, text); one PNG per word, words.txt as in IAM."""
    (root / "ascii").mkdir(parents=True)
    lines = ["# comment"]
    for word_id, text in words:
        a, b, *_ = word_id.split("-")
        folder = root / "words" / a / f"{a}-{b}"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{word_id}.png"), np.full((30, 60), 180, np.uint8))
        lines.append(f"{word_id} ok 154 1 2 3 4 5 NN {text}")
    lines.append("a99-000u-00-00 err 154 1 2 3 4 5 NN skipped")
    (root / "ascii" / "words.txt").write_text("\n".join(lines))


def test_ondb_reads_only_its_split_and_shares_the_train_charset(tmp_path):
    _write_manifest(
        tmp_path, [("a", "ab", "train"), ("b", "ba", "train"), ("c", "az", "val")]
    )

    train = Words_Dataset([load_ondb_samples(tmp_path, "train", 32, 128)])
    val = Words_Dataset(
        [load_ondb_samples(tmp_path, "val", 32, 128)], charset=train.charset
    )

    assert len(train) == 2 and len(val) == 1
    assert train.charset == ["a", "b"]
    # "z" is not in the train charset: unencodable as a target, text stays raw.
    assert val[0]["encoded"] == [0]
    assert val[0]["text"] == "az"
    assert val[0]["source"] == "ondb"
    assert train[0]["image"].shape == (32, 128)


def test_iam_split_is_by_form_and_skips_failed_segmentations(tmp_path):
    words = [(f"a0{f}-000u-00-0{w}", f"w{f}{w}") for f in range(5) for w in range(3)]
    _write_iam(tmp_path, words)

    iam = load_iam_samples(tmp_path, 32, 128, val_fraction=0.4, seed=0)

    train_forms = {n.rsplit("-", 2)[0] for n in iam["train"].names}
    val_forms = {n.rsplit("-", 2)[0] for n in iam["val"].names}
    assert len(val_forms) == 2 and len(train_forms) == 3
    assert not train_forms & val_forms
    assert len(iam["train"].images) + len(iam["val"].images) == 15  # "err" skipped
    assert iam["train"].source == "iam"


def test_combined_dataset_concatenates_sources_and_keeps_labels(tmp_path):
    ondb_dir, iam_dir = tmp_path / "ondb", tmp_path / "iam"
    ondb_dir.mkdir()
    _write_manifest(ondb_dir, [("a", "ab", "train")])
    _write_iam(iam_dir, [("a01-000u-00-00", "cd")])

    iam = load_iam_samples(iam_dir, 32, 128, val_fraction=0.0, seed=0)
    # val_fraction rounds up to at least one form, so everything is in val here.
    dataset = Words_Dataset([load_ondb_samples(ondb_dir, "train", 32, 128), iam["val"]])

    assert len(dataset) == 2
    assert [dataset[i]["source"] for i in range(2)] == ["ondb", "iam"]
    assert dataset.charset == ["a", "b", "c", "d"]
    batch = custom_collate_fn([dataset[0], dataset[1]])
    assert batch["sources"] == ["ondb", "iam"]
    assert batch["targets"].tolist() == [0, 1, 2, 3]


def test_collate_flattens_targets(tmp_path):
    _write_manifest(tmp_path, [("a", "ab", "train"), ("b", "b", "train")])
    train = Words_Dataset([load_ondb_samples(tmp_path, "train", 32, 128)])

    batch = custom_collate_fn([train[0], train[1]])

    assert batch["images"].shape == (2, 1, 32, 128)
    assert batch["target_lengths"].tolist() == [2, 1]
    assert batch["targets"].tolist() == [0, 1, 1]
