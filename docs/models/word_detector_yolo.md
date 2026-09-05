# WordDetectorYOLO model

YOLO-based word-level bounding-box detector, fine-tuned from
[YOLOv8s](https://github.com/ultralytics/ultralytics) on the IAM Handwriting
Database. Integrated into Xournal++ HTR according to the
[ADRs](../ADRs/) (in particular
[ADR 006](../ADRs/006_model_registry_and_training_environment.md) and
[ADR 007](../ADRs/007_model_demos_local_only.md)).

The source lives under
[`xournalpp_htr/training/word_detector_yolo/`](https://github.com/PellelNitram/xournalpp_htr/tree/master/xournalpp_htr/training/word_detector_yolo).

## Structure (ADR 006)

This is no longer a standalone `uv` project; it is part of the main package.

| File | Purpose | Deps |
| --- | --- | --- |
| `config.py` | Hydra structured config (single source of truth for all constants) | --- |
| `train.py` | Training entrypoint (Hydra CLI), includes dataset download + conversion | `training-word-detector-yolo` |
| `export.py` | ONNX + `config.json` export, HF Hub upload | `training-word-detector-yolo` |
| `predict.py` | Local inference from a `.pt` checkpoint | `training-word-detector-yolo` |
| `demo.py` | Local Gradio demo (run locally, not a HF Space, ADR 007) | `training-word-detector-yolo` |
| `run_training.sh` | Hyperparameter sweep | `training-word-detector-yolo` |

The HF-Hub-backed inference class lives in
`xournalpp_htr/inference_models.py` as `YOLOWordDetectorModel`.

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
uv sync --extra training-word-detector-yolo
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

Required for downloading the training dataset and (later) uploading the
exported model:

```bash
hf auth login
```

### 5. Train

Single training run (uses [Hydra](https://hydra.cc/) for configuration):

```bash
uv run python -m xournalpp_htr.training.word_detector_yolo.train \
    training.epochs=50 training.batch=16 training.lr0=0.001
```

The dataset is downloaded and converted automatically on the first run.

Show all configurable parameters and their defaults:

```bash
uv run python -m xournalpp_htr.training.word_detector_yolo.train --cfg job
```

Or run the full hyperparameter sweep:

```bash
cd xournalpp_htr/training/word_detector_yolo
bash run_training.sh
```

Results are written to `runs/detect/train_<timestamp>/`. Each run
produces `weights/best.pt`, `results.csv`, and TensorBoard logs.

Monitor training with TensorBoard (forward port 6006 if remote):

```bash
tensorboard --logdir runs/detect/ --port 6006
```

### 6. Inspect the best model

Use the Gradio demo to visually check detections:

```bash
uv run python -m xournalpp_htr.training.word_detector_yolo.demo \
    --model-path runs/detect/train_<timestamp>/weights/best.pt \
    --device auto --share
```

### 7. Export to ONNX

```bash
uv run python -m xournalpp_htr.training.word_detector_yolo.export \
    --checkpoint runs/detect/train_<timestamp>/weights/best.pt \
    --output-dir exports/
```

Produces `exports/model.onnx` and `exports/config.json`.

### 8. Upload to HuggingFace Hub

Once satisfied with the model quality:

```bash
uv run python -m xournalpp_htr.training.word_detector_yolo.export \
    --checkpoint runs/detect/train_<timestamp>/weights/best.pt \
    --output-dir exports/ --upload
```

Requires write access to `PellelNitram/xournalpp-htr-word-detector-yolo`.

## Inference

Once `model.onnx` + `config.json` are on the Hub, inference uses
`onnxruntime` only (no `ultralytics` dependency):

```python
from xournalpp_htr.inference_models import YOLOWordDetectorModel

model = YOLOWordDetectorModel.from_pretrained()
boxes = model.detect(grayscale_image)  # list[BoundingBox]
```

WordDetectorYOLO is detection-only: it produces word bounding boxes but no
transcription. The `2026-09-02_yolo_detector` benchmark pipeline pairs it
with `SimpleHTRModel` for end-to-end HTR.

## Best model

Experiment 1 baseline (`experiments/experiment1/baseline/`), exported to
ONNX and uploaded to
[PellelNitram/xournalpp-htr-word-detector-yolo](https://huggingface.co/PellelNitram/xournalpp-htr-word-detector-yolo).

Benchmark results (`2026-09-02_yolo_detector` pipeline, ONNX inference):

| Metric | Value |
|---|---|
| Precision | 73.8% |
| Recall | 80.1% |
| CER (case-sensitive) | 34.9% |
| CER (case-insensitive) | 34.4% |
| Recall × (1 − CER_ci) | 52.5% |
| Word accuracy | 39.1% |
| Predicted words | 229 |
| GT words | 211 |
| Matched | 169 |

## Experiments

<!-- Add new experiments below, newest first. -->

### 2026-09-04 -- Hydra sweep (experiment 2) and baseline re-run

- **Goal:** compare batch size (8, 16) and learning rate (0.0005, 0.001).
- **Setup:** same as initial training, Hydra config, output to
  `experiments/experiment2/`.
- **Results:** all four runs performed very similarly to the baseline.
  Experiment 1 baseline selected for deployment.
- **Selected checkpoint:**
  `experiments/experiment1/baseline/train_20260904_223247/weights/best.pt`

### 2026-09-02 -- Initial training

- **Goal:** establish a baseline with YOLOv8s fine-tuned on IAM.
- **Setup:** IAM-DB from HF Hub, 85/15 train/val split (seed 42),
  YOLOv8s pretrained, lr0=0.001, batch=16, imgsz=1024, 50 epochs,
  patience=10, AdamW optimizer, mosaic=0.5.
- **Command:** `uv run python train.py --device 0` (pre-Hydra version).
- **Results:** YOLO detector achieves the highest recall (80.1%) among
  all pipelines, at the cost of lower precision due to more predicted
  boxes (229 vs ~190).
- **Conclusion:** YOLO is a viable word detector for this task.

## Current status

Training, ONNX export, HF Hub upload, demo and lean ONNX inference (no
ultralytics dependency) are fully implemented. The model has been
benchmarked against the existing pipelines and shows the best recall and
CER. Hydra config and experiment management are in place.

## Outlook

- Confidence threshold sweep for precision/recall trade-off.
- Try YOLOv8m or larger variants.
- Increase imgsz to 1280 for better small-word detection.
- Augmentation tuning (mosaic strength, scale range).
- Validate ONNX export against PyTorch checkpoint with a Jupyter notebook.
