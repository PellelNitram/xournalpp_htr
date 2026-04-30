import pytest

from xournalpp_htr.xio import load_benchmark


@pytest.mark.slow
def test_load_benchmark_returns_pairs():
    pairs = load_benchmark()

    assert len(pairs) >= 1
    for sample in pairs:
        assert sample.xopp_path.suffix in {".xopp", ".xoj"}
        assert sample.xopp_path.is_file()
        assert sample.gt_path.suffix == ".json"
        assert sample.gt_path.stem.endswith(".gt")
        assert sample.gt_path.is_file()
        assert sample.xopp_path.stem == sample.gt_path.stem.removesuffix(".gt")
