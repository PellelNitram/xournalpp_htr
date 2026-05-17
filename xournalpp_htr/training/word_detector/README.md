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
| `demo.py` | Local sanity-check demo (no web UI / no HF Space, ADR 007) | `training-word-detector` |
| `utils.py` | Git-hash, JSON encoder, example-image list | `training-word-detector` |
| `notebooks/test_best_model.ipynb` | Inspect a trained checkpoint offline | `training-word-detector` |

Generic geometry and the map decoder/clustering/metrics live in
[`xournalpp_htr/training/shared/`](../shared/) (base deps only, importable in
the lean inference install). The HF-Hub-backed inference class lives in
[`xournalpp_htr/inference_models.py`](../../inference_models.py).

## Installation

Inference only (lean, no training deps):

```
uv add xournalpp_htr
```

Training / export / local demo:

```
uv sync --extra training-word-detector
```

## Train

```
uv run python -m xournalpp_htr.training.word_detector.train --help
```

The best checkpoint is written as `best_model.pth` (gitignored).

## Local demo (ADR 007)

A quick, offline sanity check that a trained checkpoint actually detects
words — no web UI, no HuggingFace Space, no telemetry. With no `--image` the
bundled example images are downloaded and used:

```
uv run python -m xournalpp_htr.training.word_detector.demo \
    --model-path best_model.pth --output-dir demo_output/
```

Annotated images are written to `--output-dir`. Per
[ADR 007](../../../docs/ADRs/007_model_demos_local_only.md), every contributed
model ships a local demo like this; there is no per-model HF Space.

## Export to ONNX and publish (ADR 006)

ONNX is the canonical inference artifact. Export the trained checkpoint, then
upload it to the HF Hub repo `PellelNitram/xournalpp-htr-word-detector`:

```
uv run python -m xournalpp_htr.training.word_detector.export \
    --checkpoint best_model.pth --output-dir exports/ [--upload]
```

`--upload` requires HF authentication (`huggingface-cli login` or `HF_TOKEN`)
and write access to the model repo. This produces `model.onnx` (softmax baked
into the graph) and `config.json` (pre/post-processing parameters).

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
