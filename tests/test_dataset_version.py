"""Tests for the --dataset-version / dataset_version parameter."""

from unittest.mock import patch

from scripts.run_benchmark import parse_arguments

# --- CLI parsing ---


def test_dataset_version_default_is_none():
    args = parse_arguments("")
    assert args["dataset_version"] is None


def test_dataset_version_long_flag():
    args = parse_arguments("--dataset-version v1.0")
    assert args["dataset_version"] == "v1.0"


def test_dataset_version_short_flag():
    args = parse_arguments("-d abc123")
    assert args["dataset_version"] == "abc123"


def test_dataset_version_with_full_sha():
    sha = "a" * 40
    args = parse_arguments(f"-d {sha}")
    assert args["dataset_version"] == sha


# --- load_benchmark passes revision through ---


@patch("xournalpp_htr.xio.snapshot_download")
def test_load_benchmark_passes_revision(mock_snapshot, tmp_path):
    mock_snapshot.return_value = str(tmp_path)
    (tmp_path / "data").mkdir()

    from xournalpp_htr.xio import load_benchmark

    load_benchmark(dataset_version="v2.0")

    mock_snapshot.assert_called_once_with(
        "PellelNitram/xournalpp_htr_benchmark",
        repo_type="dataset",
        revision="v2.0",
    )


@patch("xournalpp_htr.xio.snapshot_download")
def test_load_benchmark_defaults_revision_to_none(mock_snapshot, tmp_path):
    mock_snapshot.return_value = str(tmp_path)
    (tmp_path / "data").mkdir()

    from xournalpp_htr.xio import load_benchmark

    load_benchmark()

    mock_snapshot.assert_called_once_with(
        "PellelNitram/xournalpp_htr_benchmark",
        repo_type="dataset",
        revision=None,
    )
