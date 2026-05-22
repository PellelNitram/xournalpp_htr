# WordDetector model

Training, export and demo code for the WordDetector word-detection model. The
[WordDetectorNN](https://github.com/githubharald/WordDetectorNN) model was
originally created by [Harald Scheidl](https://github.com/githubharald) and is
reimplemented here with some best practices and integrated into Xournal++ HTR
according to the [ADRs](../../../docs/ADRs/) (in particular
[ADR 006](../../../docs/ADRs/006_model_registry_and_training_environment.md) and
[ADR 007](../../../docs/ADRs/007_model_demos_local_only.md)).

## Structure (ADR 006)

This is no longer a standalone `uv` project; it is part of the main package.

| File | Purpose | Deps |
| --- | --- | --- |
| `network.py` | `WordDetectorNet` architecture + training loss | `training-word-detector` |
| `dataset.py` | IAM dataset loading + ground-truth encoding | `training-word-detector` |
| `train.py` | Training entrypoint | `training-word-detector` |
| `export.py` | ONNX + `config.json` export, HF Hub upload | `training-word-detector` |
| `infer.py` | Local torch inference from a `.pth` checkpoint | `training-word-detector` |
| `demo.py` | Local Gradio demo (run locally, not a HF Space, ADR 007) | `training-word-detector` |
| `utils.py` | Git-hash, JSON encoder, example-image list | `training-word-detector` |
| `test_best_model.ipynb` | Inspect a trained checkpoint offline | `training-word-detector`, `dev` |
| `run_training.sh` | Hyperparameter sweep (grid search) | `training-word-detector` |
| `run_training.eval.sh` | Find the best model from a sweep | — |

Generic geometry and the map decoder/clustering/metrics live in
[`xournalpp_htr/training/shared/`](../shared/) (base deps only, importable in
the lean inference install). The HF-Hub-backed inference class lives in
[`xournalpp_htr/inference_models.py`](../../inference_models.py).

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

Single training run:

```bash
uv run python -m xournalpp_htr.training.word_detector.train \
    --epoch_max 200 --batch_size 32 --learning_rate 0.001
```

Or run the full hyperparameter sweep:

```bash
cd xournalpp_htr/training/word_detector
bash run_training.sh
```

Results are written to `experiments/experiment1/lr<LR>_bs<BS>/`. Each run
produces `best_model.pth`, `best_model.json` (best val F1, epoch),
TensorBoard logs in `summary_writer/`, and `args.json`.

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

## Current status

Everything from the original WordDetectorNN model is reimplemented except for
training data augmentations. Implementing those is worthwhile future work;
afterwards the bentham sample should be re-checked for correctness as it
currently fails badly.

## Outlook

- Train using data augmentations.
- Use PIL images everywhere instead of numpy to keep track of channel order.
- Revisit `PyTorchModelHubMixin` once the architecture stabilises (ADR 006).
