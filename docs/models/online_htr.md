# OnlineHTR

Stroke-based word and line recognition: a bidirectional LSTM with CTC loss
after [Carbune et al. (2020)](https://doi.org/10.1007/s10032-020-00350-4),
ported from [OnlineHTR](https://github.com/PellelNitram/OnlineHTR) into plain
PyTorch (no Lightning, no `src.*` imports). Tracked in issue #174.

Differences to the reference:

| | OnlineHTR | Here |
|---|---|---|
| Features | `(dx, dy, dt, n)` | `(dx, dy, n)`: Xournal++ stores no timestamps |
| Resampling | in normalised time | by arc length |
| CTC blank | index 0 | last class, as SimpleHTR |
| Padding | none | packed sequences |
| Data | IAM-OnDB lines | IAM-OnDB lines **and** words (cut at stroke gaps) |
| Augmentation | none | x-scale, shear, rotation |
| LR schedule | fixed | `ReduceLROnPlateau` on val CER |

Pipeline contract: [ADR 003](../ADRs/003_define_target_architecture.md),
inference format and training environment:
[ADR 006](../ADRs/006_model_registry_and_training_environment.md).

Source: `xournalpp_htr/training/online_htr/`

## Structure

| File | Purpose | Deps |
|------|---------|------|
| `features.py` | Strokes to `(dx, dy, n)`; shared by training and inference | base |
| `config.py` | Hydra structured config | base |
| `network.py` | `OnlineHTRNet`, CTC loss, greedy decoding | training |
| `dataset.py` | IAM-OnDB lines/words, writer split, augmentation | training |
| `train.py` | Training entrypoint (`@hydra.main`) | training |
| `infer.py` | Local torch inference from `.pth` | training |
| `export.py` | ONNX export + HF Hub upload | training |
| `utils.py` | Device, git hash, edit distance | training |
| `run_training.sh` | Experiments | shell |

## GPU training setup

```bash
git clone https://github.com/PellelNitram/xournalpp_htr.git
cd xournalpp_htr
bash INSTALL_LINUX.sh
export HF_TOKEN=...   # the IAM-OnDB dataset repo is private
bash xournalpp_htr/training/online_htr/run_training.sh
```

The script resolves IAM-OnDB from HF Hub, trains and writes
`experiments/experiment1/baseline/` with `best_model.pth`, `charset.json`,
`config.yaml`, `best_model.json` and TensorBoard logs. Export with:

```bash
uv run --extra training-online-htr python -m xournalpp_htr.training.online_htr.export \
    --checkpoint experiments/experiment1/baseline/best_model.pth \
    --output-dir experiments/experiment1/baseline/export
```

## Inference

Not wired into `inference_models.py` yet.

## Best model

None trained yet.

## Experiments

None yet.

## Current status

Training and ONNX export are implemented and tested on synthetic data. No
inference class, demo or pipeline integration yet.

## Outlook

- `OnlineHTRModel` in `inference_models.py` and a benchmark run behind
  `WordDetectorOnline`.
- Local Gradio demo (ADR 007).
- Beam search with `shared/ctc_decoding.py`.
