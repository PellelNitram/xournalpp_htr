---
title: Xournal++ HTR WordDetectorNN
emoji: 📄
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.44.1
app_file: demo.py
pinned: false
---

# WordDetector model

[🤗 demo](https://huggingface.co/spaces/PellelNitram/xournalpp_htr_WordDetectorNN)

Training, export and demo code for the WordDetector word-detection model. The
[WordDetectorNN](https://github.com/githubharald/WordDetectorNN) model was
originally created by [Harald Scheidl](https://github.com/githubharald) and is
reimplemented here with some best practices and integrated into Xournal++ HTR
according to the [ADRs](../../../docs/ADRs/) (in particular
[ADR 006](../../../docs/ADRs/006_model_registry_and_training_environment.md)).

## Structure (ADR 006)

This is no longer a standalone `uv` project; it is part of the main package.

| File | Purpose | Deps |
| --- | --- | --- |
| `network.py` | `WordDetectorNet` architecture + training loss | `training-word-detector` |
| `dataset.py` | IAM dataset loading + ground-truth encoding | `training-word-detector` |
| `train.py` | Training entrypoint | `training-word-detector` |
| `export.py` | ONNX + `config.json` export, HF Hub upload | `training-word-detector` |
| `infer.py` | Local torch inference from a `.pth` checkpoint | `training-word-detector` |
| `demo.py` / `events.py` | Gradio HF Space demo + Supabase logging | `training-word-detector`, `hf` |
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

Training / export / demo:

```
uv sync --extra training-word-detector        # add --extra hf for the demo
```

## Train

```
uv run python -m xournalpp_htr.training.word_detector.train --help
```

The best checkpoint is written as `best_model.pth` (gitignored).

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

## Deployment as Hugging Face Space

The model is deployed as the HF Gradio Space linked above. Deployment is
currently a manual file copy; automating it via a Docker HF Space is future
work.

## Supabase database commands

Create the events table:

```sql
create table word_detector_net.hf_space_events (
  id bigserial primary key,
  timestamp timestamptz not null,
  demo boolean not null,
  uuid text not null,
  donate_data bool not null,
  contains_image bool not null
);
```

Create bucket:

```
WordDetectorNN_hf_space_images
```

Configure table:

```
-- Schema access
grant usage on schema word_detector_net to service_role;

-- Table access
grant insert, select on table word_detector_net.hf_space_events to service_role;

-- Sequence access for autoincrement IDs
grant usage, select, update on sequence word_detector_net.hf_space_events_id_seq to service_role;
```

## Outlook

- Train using data augmentations.
- Use PIL images everywhere instead of numpy to keep track of channel order.
- Revisit `PyTorchModelHubMixin` once the architecture stabilises (ADR 006).
