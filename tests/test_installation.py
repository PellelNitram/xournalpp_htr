# ruff: noqa: F401

import pytest


@pytest.mark.installation
def test_htr_pipeline_package_installation() -> None:
    """Test if code from `htr_pipeline` package can be imported."""
    from htr_pipeline import DetectorConfig, LineClusteringConfig, read_page


@pytest.mark.installation
def test_xournalpp_htr_package_installation() -> None:
    """Test if code from `xournalpp_htr` package can be imported."""
    from xournalpp_htr.documents import XournalDocument, XournalppDocument
    from xournalpp_htr.utils import export_to_pdf_with_xournalpp
