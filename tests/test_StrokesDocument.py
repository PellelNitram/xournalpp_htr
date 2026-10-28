import cv2
import numpy as np

from xournalpp_htr.documents import StrokesDocument


def test_strokes_document_renders_through_save_page_as_image(tmp_path):
    x = np.linspace(10, 90, 20)
    document = StrokesDocument([(x, np.full_like(x, 30.0))], width=100, height=60)

    out = document.save_page_as_image(0, tmp_path / "page.png", False, dpi=150)

    image = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (125, 208)  # 100 x 60 pt at 150 DPI
    ys, xs = np.where(image < 128)
    # The horizontal line sits at y = 30 pt -> 62.5 px and spans x = 10..90 pt.
    assert abs(ys.mean() - 30 * 150 / 72) < 2
    assert abs(xs.min() - 10 * 150 / 72) < 4
    assert abs(xs.max() - 90 * 150 / 72) < 4
