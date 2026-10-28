import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx", reason="needed by torch.onnx.export")
omegaconf = pytest.importorskip("omegaconf")

from xournalpp_htr.inference_models import SimpleHTRModel  # noqa: E402
from xournalpp_htr.training.simple_htr_ondb.config import ModelConfig  # noqa: E402
from xournalpp_htr.training.simple_htr_ondb.export import export  # noqa: E402
from xournalpp_htr.training.simple_htr_ondb.network import SimpleHTRNet  # noqa: E402


def test_export_loads_with_from_local_and_matches_torch(tmp_path):
    cfg = ModelConfig()
    charset = list("abc")
    net = SimpleHTRNet(num_classes=len(charset) + 1, cfg=cfg).eval()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(net.state_dict(), run_dir / "best_model.pth")
    (run_dir / "charset.json").write_text(json.dumps(charset))
    omegaconf.OmegaConf.save({"model": asdict(cfg)}, run_dir / "config.yaml")

    paths = export(run_dir / "best_model.pth", tmp_path / "export")
    assert paths["onnx"].exists() and paths["config"].exists()
    assert json.loads(paths["config"].read_text())["model_name"] == "simple_htr_ondb"

    model = SimpleHTRModel.from_local(tmp_path / "export")
    assert model.revision == "local"

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(50, 120), dtype=np.uint8)
    assert isinstance(model.recognize(img), str)

    # The ONNX graph must reproduce the PyTorch network on the same input.
    net_input = model.preprocess(img).astype(np.float32) / 255.0 - 0.5
    with torch.no_grad():
        expected = net(torch.from_numpy(net_input[None, None]))[:, 0, :].numpy()
    np.testing.assert_allclose(model._compute_log_probs(img), expected, atol=1e-4)
