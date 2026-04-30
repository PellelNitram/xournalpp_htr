from pathlib import Path

import pytest

from xournalpp_htr.documents import get_document
from xournalpp_htr.xio import load_examples


def _find_example(file_paths: list[str], name: str) -> Path:
    matches = [Path(p) for p in file_paths if Path(p).name == name]
    assert matches, f"Example file '{name}' not found in downloaded dataset"
    return matches[0]


@pytest.fixture(scope="module")
def example_files() -> list[str]:
    return load_examples()


@pytest.mark.data
@pytest.mark.parametrize(
    "filename",
    [
        "empty.xopp",
        "empty.xoj",
        "empty_and_not_empty.xopp",
        "first_upload.xoj",
    ],
)
def test_get_document_does_not_crash_on_empty_file(
    filename: str, example_files: list[str]
) -> None:
    """Test that `get_document` does not crash when processing an empty Xournal++ file."""
    path = _find_example(example_files, filename)
    get_document(path)
