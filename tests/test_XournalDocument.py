from pathlib import Path

from xournalpp_htr.documents import XournalppDocument


def test_XournalppDocument(get_path_to_minimal_test_data: Path):
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
