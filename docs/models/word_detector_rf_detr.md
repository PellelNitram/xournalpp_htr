# WordDetectorRFDETR model

RF-DETR-based word-level bounding-box detector, fine-tuned from
[RF-DETR](https://github.com/roboflow/rf-detr) on the IAM Handwriting
Database. RF-DETR is a DETR-style transformer detector, so decoding is
NMS-free — which is the main reason to try it next to the YOLO detector on
densely packed text lines. Integrated into Xournal++ HTR according to the
[ADRs](../ADRs/) (in particular
[ADR 006](../ADRs/006_model_registry_and_training_environment.md) and
[ADR 007](../ADRs/007_model_demos_local_only.md)).

The source lives under
[`xournalpp_htr/training/word_detector_rf_detr/`](https://github.com/PellelNitram/xournalpp_htr/tree/master/xournalpp_htr/training/word_detector_rf_detr).

## Structure (ADR 006)

| File | Purpose | Deps |
| --- | --- | --- |
| `config.py` | Hydra structured config (single source of truth for all constants) | --- |
| `train.py` | Training entrypoint (Hydra CLI), includes dataset download + COCO conversion | `training-word-detector-rf-detr` |
| `export.py` | ONNX + `config.json` export, HF Hub upload | `training-word-detector-rf-detr` |
| `predict.py` | Local inference from a `.pth` checkpoint | `training-word-detector-rf-detr` |
| `demo.py` | Local Gradio demo (run locally, not a HF Space, ADR 007) | `training-word-detector-rf-detr` |
| `run_training.sh` | Hyperparameter sweep | `training-word-detector-rf-detr` |

The HF-Hub-backed inference class will live in
`xournalpp_htr/inference_models.py` as `RFDETRWordDetectorModel`. It is **not
implemented yet** — it is written once the ONNX export has been validated,
because the post-processing depends on the exact output signature of the
export.

## GPU training setup (step-by-step)

Prerequisites: a Linux machine with an NVIDIA GPU, CUDA drivers installed
(`nvidia-smi` should work), and `uv` installed (`pip install uv`).

### 1. Clone and install the base package

```bash
git clone https://github.com/PellelNitram/xournalpp_htr.git
cd xournalpp_htr
bash INSTALL_LINUX.sh
```

### 2. Install the training extra (with CUDA PyTorch)

```bash
uv sync --extra training-word-detector-rf-detr
```

Verify GPU access:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Verify the installation

```bash
make tests-not-slow
```

### 4. Authenticate with HuggingFace

Required for downloading the training dataset, for the pretrained RF-DETR
backbone weights, and (later) for uploading the exported model:

```bash
hf auth login
```

### 5. Train

Single training run (uses [Hydra](https://hydra.cc/) for configuration):

```bash
uv run python -m xournalpp_htr.training.word_detector_rf_detr.train \
    training.epochs=50 training.batch_size=4 training.lr=1e-4
```

The dataset is downloaded and converted to COCO format automatically on the
first run, into `dataset/train/` and `dataset/valid/` with an
`_annotations.coco.json` per split.

Show all configurable parameters and their defaults:

```bash
uv run python -m xournalpp_htr.training.word_detector_rf_detr.train --cfg job
```

Or run the full hyperparameter sweep:

```bash
cd xournalpp_htr/training/word_detector_rf_detr
bash run_training.sh
```

Two knobs differ from the YOLO detector and are worth knowing:

- **Effective batch size** is `training.batch_size * training.grad_accum_steps`.
  RF-DETR is tuned for a total of 16; if you lower `batch_size` to fit in GPU
  memory, raise `grad_accum_steps` to compensate.
- **`model.resolution` must be divisible by 56** (a constraint of RF-DETR's
  positional embeddings). The default 1008 = 56 × 18 is the closest analogue
  to the YOLO detector's `imgsz=1024`. `train.py` validates this and fails
  early with a clear message.

Results are written to `outputs/train_<timestamp>/`. Each run produces
`checkpoint_best_total.pth`, `checkpoint_best_ema.pth`, periodic checkpoints
and TensorBoard logs under `tb/`.

Monitor training with TensorBoard (forward port 6006 if remote):

```bash
tensorboard --logdir outputs/ --port 6006
```

### 6. Inspect the best model

Use the Gradio demo to visually check detections:

```bash
uv run python -m xournalpp_htr.training.word_detector_rf_detr.demo \
    --model-path outputs/train_<timestamp>/checkpoint_best_total.pth \
    --share
```

### 7. Export to ONNX

```bash
uv run python -m xournalpp_htr.training.word_detector_rf_detr.export \
    --checkpoint outputs/train_<timestamp>/checkpoint_best_total.pth \
    --output-dir exports/
```

Produces `exports/model.onnx` and `exports/config.json`. Pass `--variant` and
`--resolution` if the checkpoint was trained with non-default values.

### 8. Validate the export, then upload to HuggingFace Hub

Validate the ONNX export against the PyTorch checkpoint in a notebook first
(see the conventions in [the models overview](index.md)), then:

```bash
uv run python -m xournalpp_htr.training.word_detector_rf_detr.export \
    --checkpoint outputs/train_<timestamp>/checkpoint_best_total.pth \
    --output-dir exports/ --upload
```

Requires write access to `PellelNitram/xournalpp-htr-word-detector-rf-detr`.

## Inference

Not available yet — `RFDETRWordDetectorModel` still has to be added to
`xournalpp_htr/inference_models.py` once a model has been exported and
validated. It will follow the same shape as the other detectors:

```python
from xournalpp_htr.inference_models import RFDETRWordDetectorModel

model = RFDETRWordDetectorModel.from_pretrained()
boxes = model.detect(grayscale_image)  # list[BoundingBox]
```

Like WordDetectorYOLO, this model is detection-only: it produces word
bounding boxes but no transcription, and is meant to be paired with
`SimpleHTRModel` in a benchmark pipeline.

## Best model

None yet — no training run has been performed.

## Experiments

<!-- Add new experiments below, newest first. -->

None yet.

## Current status

Scaffolding only. Training, export, predict and demo scripts are in place and
follow the conventions in [the models overview](index.md), but **nothing has
been trained or run end-to-end yet**. In particular:

- The RF-DETR API calls in `train.py`, `export.py` and `predict.py` are
  written against the `rfdetr` package's documented interface and have not
  been executed against an installed version.
- `RFDETRWordDetectorModel` does not exist in `xournalpp_htr.inference_models`.
- The IAM XML parsing in `train.py` is duplicated from the YOLO detector's
  `train.py`; it is a candidate for `xournalpp_htr/training/shared/`.

## Outlook

- First baseline run to confirm the pipeline works end to end.
- Compare recall against WordDetectorYOLO (80.1%) — recall is the metric this
  detector is optimised for.
- Confidence threshold sweep for the precision/recall trade-off.
- Try `model.variant=large` and higher `model.resolution`.
- Lift the shared IAM XML parsing into `xournalpp_htr/training/shared/`.
