from pathlib import Path

import pytest

from xournalpp_htr.documents import XournalppDocument, get_document


def test_XournalppDocument(get_path_to_minimal_test_data: Path) -> None:
    """Tests `XournalppDocument` and thereby its `load_data` function."""
    xpp_document = XournalppDocument(get_path_to_minimal_test_data)
    assert xpp_document.path == get_path_to_minimal_test_data
    assert len(xpp_document.pages) == 1
    assert xpp_document.DPI == 72
    page = xpp_document.pages[0]
    assert page.meta_data["width"] == "612"
    assert page.meta_data["height"] == "792"
    assert page.background["type"] == "solid"
    assert page.background["color"] == "#ffffffff"
    assert page.background["style"] == "lined"
    assert len(page.layers) == 1
    layer = page.layers[0]
    assert len(layer.strokes) == 85


def test_get_document_xopp(get_path_to_minimal_test_data: Path) -> None:
    """Tests `get_document` function for a `xopp` file."""
    xpp_document = XournalppDocument(get_path_to_minimal_test_data)
    document = get_document(get_path_to_minimal_test_data)
    assert xpp_document.path == document.path == get_path_to_minimal_test_data
    assert len(xpp_document.pages) == len(document.pages) == 1
    assert xpp_document.DPI == document.DPI
    assert xpp_document.pages[0].meta_data == document.pages[0].meta_data
    assert xpp_document.pages[0].background == document.pages[0].background
    assert len(xpp_document.pages[0].layers) == len(document.pages[0].layers) == 1
    assert (
        len(xpp_document.pages[0].layers[0].strokes)
        == len(document.pages[0].layers[0].strokes)
        == 85
    )


def test_get_document_unsupported() -> None:
    """Tests `get_document` function for an unsupported file extension."""
    with pytest.raises(NotImplementedError):
        get_document(Path("/a/file/that/is.unsupported"))


def test_get_min_max_coordintes_per_page(get_path_to_minimal_test_data: Path) -> None:
    """
    Regression test of `Document.get_min_max_coordintes_per_page` function.
    """
    xpp_document = XournalppDocument(get_path_to_minimal_test_data)

    result = xpp_document.get_min_max_coordintes_per_page()

    expected_result = {
        0: {
            "min_x": 110.41164,
            "min_y": 127.30273,
            "max_x": 573.78978,
            "max_y": 516.39615,
        }
    }

    assert result == expected_result
