# WordDetector model

Training, export and demo code for the WordDetector word-detection model. The
[WordDetectorNN](https://github.com/githubharald/WordDetectorNN) model was
originally created by [Harald Scheidl](https://github.com/githubharald) and is
reimplemented here with some best practices and integrated into Xournal++ HTR
according to the [ADRs](../ADRs/) (in particular
[ADR 006](../ADRs/006_model_registry_and_training_environment.md) and
[ADR 007](../ADRs/007_model_demos_local_only.md)).

The source lives under
[`xournalpp_htr/training/word_detector/`](https://github.com/PellelNitram/xournalpp_htr/tree/master/xournalpp_htr/training/word_detector).

## Structure (ADR 006)

This is no longer a standalone `uv` project; it is part of the main package.

| File | Purpose | Deps |
| --- | --- | --- |
| `config.py` | Hydra structured config (single source of truth for all constants) | — |
| `network.py` | `WordDetectorNet` architecture + training loss | `training-word-detector` |
| `dataset.py` | IAM dataset loading + ground-truth encoding | `training-word-detector` |
| `train.py` | Training entrypoint (Hydra CLI) | `training-word-detector` |
| `export.py` | ONNX + `config.json` export, HF Hub upload | `training-word-detector` |
| `infer.py` | Local torch inference from a `.pth` checkpoint | `training-word-detector` |
| `demo.py` | Local Gradio demo (run locally, not a HF Space, ADR 007) | `training-word-detector` |
| `utils.py` | Git-hash, JSON encoder, example-image list | `training-word-detector` |
| `test_best_model.ipynb` | Inspect a trained checkpoint offline | `training-word-detector`, `dev` |
| `run_training.sh` | Hyperparameter sweep (grid search) | `training-word-detector` |
| `run_training.eval.sh` | Find the best model from a sweep | — |

Generic geometry and the map decoder/clustering/metrics live in
`xournalpp_htr/training/shared/` (base deps only, importable in the lean
inference install). The HF-Hub-backed inference class lives in
`xournalpp_htr/inference_models.py`.

## GPU training setup (step-by-step)

Prerequisites: a Linux machine with an NVIDIA GPU, CUDA drivers installed
(`nvidia-smi` should work), and `uv` installed (`pip install uv`).

### 1. Clone and install the base package

```bash
git clone https://github.com/PellelNitram/xournalpp_htr.git
cd xournalpp_htr
bash INSTALL_LINUX.sh
```

The install script downloads the HTRPipeline models via `wget` from Dropbox.
If this fails (e.g. corporate proxy blocking SSL), download `models.zip` on
another machine and copy it into
`external/htr_pipeline/HTRPipeline/htr_pipeline/models/`, then `unzip -o models.zip`.
After that, copy the ONNX/JSON files into the venv:

```bash
mkdir -p .venv/lib/python3.11/site-packages/htr_pipeline/models/
cp external/htr_pipeline/HTRPipeline/htr_pipeline/models/*.onnx \
   external/htr_pipeline/HTRPipeline/htr_pipeline/models/*.json \
   .venv/lib/python3.11/site-packages/htr_pipeline/models/
```

### 2. Install the training extra (with CUDA PyTorch)

```bash
uv sync --extra training-word-detector
```

This installs PyTorch with CUDA support (cu128 index configured in
`pyproject.toml`). Verify GPU access:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If the CUDA version doesn't match your driver, update the `pytorch-cu128`
index URL in `pyproject.toml` to the appropriate version (e.g. `cu121`,
`cu124`) and re-run `uv sync --extra training-word-detector`.

### 3. Verify the installation

```bash
make tests-not-slow
```

All tests should pass except `test_run_htr::test_main` which requires the
`xournalpp` desktop application (not needed for training). No datasets are
required for this step.

### 4. Authenticate with HuggingFace

Required for downloading the training dataset and (later) uploading the
exported model:

```bash
hf auth login
```

### 5. Download the training dataset

```bash
hf download PellelNitram/xournalpp_htr_IAM_DB --repo-type dataset
```

The first run caches the dataset under `~/.cache/huggingface/`. Subsequent
runs resolve the cache instantly.

### 6. Train

Single training run (uses [Hydra](https://hydra.cc/) for configuration):

```bash
uv run python -m xournalpp_htr.training.word_detector.train \
    training.epoch_max=200 training.batch_size=32 training.learning_rate=0.001
```

To enable train-time data augmentations (off by default):

```bash
uv run python -m xournalpp_htr.training.word_detector.train \
    augmentation.enabled=true training.epoch_max=200
```

Show all configurable parameters and their defaults:

```bash
uv run python -m xournalpp_htr.training.word_detector.train --cfg job
```

Or run the full hyperparameter sweep:

```bash
cd xournalpp_htr/training/word_detector
bash run_training.sh
```

Results are written to `experiments/experiment1/lr<LR>_bs<BS>/`. Each run
produces `best_model.pth`, `best_model.json` (best val F1, epoch),
TensorBoard logs in `summary_writer/`, and `config.yaml`.

The script also contains `experiment2`, an augmentation ablation study
that compares `augmentation.enabled=false` vs `true` across three data
splits. Results go to `experiments/experiment2/aug<BOOL>_seed<SEED>/`.

Monitor training with TensorBoard (forward port 6006 if remote):

```bash
tensorboard --logdir experiments/ --port 6006
```

The first training run builds a `dataset_cache.pickle` from the HF-cached
raw files. Subsequent runs load directly from this pickle, skipping
both the HF validation and image preprocessing.

### 7. Evaluate the sweep

```bash
bash run_training.eval.sh
```

Reports the F1 score for each completed run and prints the best model path.

### 8. Inspect the best model

Use the best model path from the previous step. Visually check it with the
Gradio demo:

```bash
uv run python -m xournalpp_htr.training.word_detector.demo \
    --model-path experiments/experiment1/<best_run>/best_model.pth \
    --device auto --share
```

`--share` exposes a temporary public URL (useful on headless machines).

### 9. Export to ONNX

```bash
uv run python -m xournalpp_htr.training.word_detector.export \
    --checkpoint experiments/experiment1/<best_run>/best_model.pth \
    --output-dir exports/
```

Produces `exports/model.onnx` and `exports/config.json`.

### 10. Validate the ONNX export

Run the notebook to compare PyTorch and ONNX predictions side by side.
The notebook expects `best_model.pth` in the `word_detector/` directory:

```bash
cp experiments/experiment1/<best_run>/best_model.pth best_model.pth
uv sync --extra dev  # adds jupyter
uv run jupyter nbconvert --to notebook --execute test_best_model.ipynb \
    --output test_best_model_executed.ipynb
```

### 11. Upload to HuggingFace Hub

Once satisfied with the model quality:

```bash
uv run python -m xournalpp_htr.training.word_detector.export \
    --checkpoint experiments/experiment1/<best_run>/best_model.pth \
    --output-dir exports/ --upload
```

Requires write access to `PellelNitram/xournalpp-htr-word-detector`.

## Inference

Once `model.onnx` + `config.json` are on the Hub, inference is uniform across
all custom models (no `transformers` dependency):

```python
from xournalpp_htr.inference_models import WordDetectorModel

model = WordDetectorModel.from_pretrained()           # or revision="v1.2.0"
boxes = model.detect(grayscale_image)                 # list[BoundingBox]
```

WordDetector is detection-only: it produces word bounding boxes but no
transcription/labels. It is therefore **not** exposed as a `compute_predictions`
pipeline (ADR 003), which contracts word-level boxes *and* transcriptions. The
`WordDetectorModel` class is the integration point; wiring it into a full
pipeline waits on a recognition stage that adds labels.

## Experiments

A log of training experiments run on this model. Each entry should capture
what was tried, why, and what the outcome was, so that future work can
reproduce or build on the result.

Suggested entry template:

```
### <date> — <short title>

- **Goal:** what question this experiment is meant to answer.
- **Setup:** dataset split, config overrides, code revision (commit hash).
- **Command:** the exact training/eval command used.
- **Results:** key metrics (val F1, etc.), path to artefacts under
  `experiments/`, TensorBoard run name.
- **Conclusion:** what was learned and what to try next.
```

<!-- Add new experiments below, newest first. -->

### 2026-06-01 — Experiment 3: augmentation under original training regime

- **Goal:** does augmentation help when training closer to the original
  WordDetectorNN regime (bs=10, unbounded epochs with early stopping)?
  Experiment 2 showed augmentation hurting under our regime (bs=32,
  epoch_max=100), but the original repo always uses augmentation with a
  much smaller batch size and no epoch cap.
- **Setup:** IAM-DB, 80/20 random split, lr=0.001, bs=10, epoch_max=10000
  (effectively unbounded, relies on patience_max=50), input 448x448.
  Augmentation on vs off, 3 seeds each (42, 43, 44). Code revision
  `948f48f`.
- **Command:** `bash run_training.sh` (experiment3 function).
- **Results:**

  | Augmentation | Seed 42 | Seed 43 | Seed 44 | Mean F1 |
  |---|---|---|---|---|
  | Off | 0.8750 | 0.8786 | 0.8799 | **0.8779** |
  | On | 0.8896 | 0.8819 | 0.8787 | **0.8834** |

  Augmented runs trained longer before early stopping (139–416 epochs vs
  95–165 epochs), indicating augmentation adds useful diversity the model
  can exploit given enough training time.

  Artefacts: `experiments/experiment3/aug{true,false}_seed{42,43,44}/`.

- **Conclusion:** under the original-like regime (bs=10, unbounded epochs),
  augmentation helps (+0.55 pp mean F1). Both conditions also outperform
  experiment 2 overall (mean F1 ~0.88 vs ~0.86), confirming that the
  smaller batch size with longer training is beneficial. The recommended
  final training configuration is **bs=10, augmentation on, unbounded
  epochs with early stopping (patience 50)**.

### 2026-05-28 — Experiment 2: augmentation ablation

- **Goal:** does train-time data augmentation improve word detector performance?
- **Setup:** IAM-DB, 80/20 random split, lr=0.001, bs=32, epoch_max=100,
  patience_max=50, input 448x448. Augmentation on vs off, 3 seeds each
  (42, 43, 44). Code revision `58a26e4`.
- **Command:** `bash run_training.sh` (experiment2 function).
- **Results:**

  | Augmentation | Seed 42 | Seed 43 | Seed 44 | Mean F1 |
  |---|---|---|---|---|
  | Off | 0.8596 | 0.8681 | 0.8590 | **0.8622** |
  | On | 0.8571 | 0.8574 | 0.8609 | **0.8584** |

  Artefacts: `experiments/experiment2/aug{true,false}_seed{42,43,44}/`.

- **Conclusion:** augmentation slightly hurts performance (~0.4 pp lower
  mean F1). However, our training regime differs from the
  [original WordDetectorNN](https://github.com/githubharald/WordDetectorNN)
  (bs=10, unbounded epochs with early stopping, first-20-sample val split,
  350x350 input). Experiment 3 will test augmentation under an
  original-like regime (bs=10, unbounded epochs) to see if augmentation
  helps there.

## Current status

Everything from the original WordDetectorNN model is reimplemented, including
train-time data augmentations (geometric and photometric, matching
githubharald/WordDetectorNN). Augmentations are off by default and can be
enabled via `augmentation.enabled=true`. The bentham sample should be
re-checked for correctness as it currently fails badly.

## Outlook

- Use PIL images everywhere instead of numpy to keep track of channel order.
- Revisit `PyTorchModelHubMixin` once the architecture stabilises (ADR 006).
