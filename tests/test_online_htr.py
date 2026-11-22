import json
from dataclasses import asdict

import numpy as np
import pytest

from xournalpp_htr.training.online_htr.features import strokes_to_features

torch = pytest.importorskip("torch")
pytest.importorskip("onnx", reason="needed by torch.onnx.export")
omegaconf = pytest.importorskip("omegaconf")

from xournalpp_htr.training.online_htr.config import (  # noqa: E402
    DataConfig,
    FeatureConfig,
    ModelConfig,
)
from xournalpp_htr.training.online_htr.dataset import (  # noqa: E402
    OnlineHTRDataset,
    _samples_of_line,
    custom_collate_fn,
)
from xournalpp_htr.training.online_htr.export import export  # noqa: E402
from xournalpp_htr.training.online_htr.network import (  # noqa: E402
    OnlineHTRNet,
    compute_ctc_loss,
    greedy_decode,
)


def _strokes():
    t = np.linspace(0, 1, 30)
    return [(t * 3, np.sin(t * 6)), (t * 3 + 4, np.cos(t * 6))]


def test_features_shape_and_stroke_starts():
    feats = strokes_to_features(_strokes())
    assert feats.dtype == np.float32 and feats.shape[1] == 3
    assert feats[:, 2].sum() == 2  # one start per stroke
    assert feats[0, 2] == 1 and feats[0, 0] == 0 and feats[0, 1] == 0


def test_features_are_scaled_to_unit_height():
    feats = strokes_to_features(_strokes())
    y = np.cumsum(feats[:, 1])
    assert y.max() - y.min() == pytest.approx(
        1.0, abs=0.01
    )  # resampling may skip an extremum


def test_features_resample_by_arc_length():
    strokes = [(np.array([0.0, 0.001, 1.0]), np.array([0.0, 0.0, 1.0]))]
    steps = np.hypot(*strokes_to_features(strokes, 20.0)[1:, :2].T)
    assert steps.std() < 0.2 * steps.mean()  # evenly spaced, unlike the input


@pytest.mark.parametrize(
    "strokes, n_points",
    [
        ([], 0),
        ([(np.array([1.0]), np.array([2.0]))], 1),  # a dot
        ([(np.array([1.0, 1.0]), np.array([2.0, 2.0]))], 1),  # coinciding points
        ([(np.array([0.0, 5.0]), np.array([1.0, 1.0]))], None),  # flat line
    ],
)
def test_features_degenerate_inputs(strokes, n_points):
    feats = strokes_to_features(strokes)
    assert np.isfinite(feats).all()
    if n_points is not None:
        assert len(feats) == n_points


def test_packed_batch_matches_single_sample():
    torch.manual_seed(0)
    net = OnlineHTRNet(num_classes=5, cfg=ModelConfig()).eval()
    a, b = torch.randn(12, 3), torch.randn(7, 3)
    batch = torch.nn.utils.rnn.pad_sequence([a, b])
    out = net(batch, torch.tensor([12, 7]))
    with torch.no_grad():
        np.testing.assert_allclose(
            out[:7, 1].detach(), net(b[:, None])[:, 0].detach(), atol=1e-5
        )


def test_greedy_decode_collapses_repeats_and_blanks():
    charset = ["a", "b"]  # blank is index 2
    path = [0, 0, 2, 0, 1, 1]
    log_probs = torch.full((len(path), 1, 3), -10.0)
    for t, c in enumerate(path):
        log_probs[t, 0, c] = 0.0
    assert greedy_decode(log_probs, charset) == ["aab"]


def test_collate_and_loss_run():
    charset = list("abc")
    entry = {
        "sample_name": "a01-000u-01",
        "label": "ab c",
        "stroke_nr": [0] * 30 + [1] * 30,
        "x": np.concatenate([_strokes()[0][0], _strokes()[1][0]]),
        "y": -np.concatenate([_strokes()[0][1], _strokes()[1][1]]),
    }
    samples = _samples_of_line(entry, "w1", DataConfig(level="line"))
    assert [s.text for s in samples] == ["ab c"]
    ds = OnlineHTRDataset(samples * 3, charset, 20.0)
    batch = custom_collate_fn([ds[i] for i in range(3)])
    net = OnlineHTRNet(num_classes=len(charset) + 1, cfg=ModelConfig())
    log_probs = net(batch["ink"], batch["ink_lengths"])
    loss = compute_ctc_loss(
        log_probs,
        batch["targets"],
        batch["ink_lengths"],
        batch["target_lengths"],
        blank=len(charset),
    )
    assert torch.isfinite(loss)


def test_export_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    cfg = ModelConfig()
    charset = list("abc")
    net = OnlineHTRNet(num_classes=len(charset) + 1, cfg=cfg).eval()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(net.state_dict(), run_dir / "best_model.pth")
    (run_dir / "charset.json").write_text(json.dumps(charset))
    omegaconf.OmegaConf.save(
        {"model": asdict(cfg), "features": asdict(FeatureConfig())},
        run_dir / "config.yaml",
    )

    paths = export(run_dir / "best_model.pth", tmp_path / "export")
    assert json.loads(paths["config"].read_text())["model_name"] == "online_htr"

    feats = strokes_to_features(_strokes())[:, None, :]
    session = ort.InferenceSession(str(paths["onnx"]))
    (onnx_out,) = session.run(None, {"ink": feats})
    with torch.no_grad():
        np.testing.assert_allclose(
            onnx_out, net(torch.from_numpy(feats)).numpy(), atol=1e-4
        )
