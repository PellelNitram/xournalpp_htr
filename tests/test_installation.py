import pytest


@pytest.mark.installation
def test_htr_pipeline_package_installation():
    """Test if code from `htr_pipeline` package can be imported."""
    from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig

@pytest.mark.installation
def test_xournalpp_htr_package_installation():
    """Test if code from `xournalpp_htr` package can be imported."""
    from xournalpp_htr.documents import XournalDocument
    from xournalpp_htr.documents import XournalppDocument
    from xournalpp_htr.utils import export_to_pdf_with_xournalpp