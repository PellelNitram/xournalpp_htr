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

Or run all experiments via `run_training.sh`, which writes each run under
`experiments/<experiment>/<run>/` together with its `train.log`:

```bash
bash xournalpp_htr/training/word_detector_rf_detr/run_training.sh
```

The script `cd`s to its own directory, so it can be invoked from anywhere. Run
it under `tmux` — it is the baseline plus a three-point learning-rate sweep,
so hours, not minutes. To run a single configuration instead, call `train.py`
directly with Hydra overrides as shown above.

Two knobs differ from the YOLO detector and are worth knowing:

- **Effective batch size** is `training.batch_size * training.grad_accum_steps`.
  RF-DETR is tuned for a total of 16, so keep that product fixed when changing
  either. The defaults are 8 × 2: measured at 1024px on a 23GB L4,
  `batch_size=2` used only 5.5GB, so 8 leaves headroom and shortens epochs.
- **`model.variant` depends on your installed `rfdetr`.** Newer releases
  dropped `base` in favour of `nano`/`small`/`medium`/`large`; the default
  here is `medium`, the closest successor to the old `base`. List what your
  install actually ships with:

  ```bash
  uv run python -c "from xournalpp_htr.training.word_detector_rf_detr.model_factory import available_variants; print(available_variants())"
  ```

  The `seg*` and `keypointpreview` entries in that list are segmentation and
  keypoint models, not word-box detectors.
- **`model.resolution` must be divisible by `patch_size * num_windows`**,
  which is 32 for every detection variant in rfdetr 1.10.1. The default is
  1024 — the same input resolution as the YOLO detector, so the two are
  compared like for like, and well above the variant defaults (nano 384,
  small 512, medium 576, large 704) because IAM words are small.
  `model_factory.validate_resolution` reads the divisor off the model and
  fails early, naming the nearest valid value.

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

Inference uses `onnxruntime` only (no `rfdetr` dependency):

```python
from xournalpp_htr.inference_models import RFDETRWordDetectorModel

model = RFDETRWordDetectorModel.from_pretrained()
boxes = model.detect(grayscale_image)                  # list[BoundingBox]
boxes = model.detect(grayscale_image, threshold=0.25)  # lower = more boxes
```

Decoding mirrors `rfdetr.models.postprocess.PostProcess`: sigmoid over the
class logits, flatten across (query, class), stable top-300 selection, then
`cxcywh` → `xyxy` scaled by the original image size. There is **no NMS** —
RF-DETR's Hungarian matching makes each of its 300 object queries responsible
for at most one object, unlike the YOLO detector which needs explicit
suppression. The square resize is deliberate and must not be replaced with
letterboxing: boxes are normalised to the resized frame and mapped back by
multiplying with the original width and height, so padding would break the
mapping.

Like WordDetectorYOLO, this model is detection-only: it produces word
bounding boxes but no transcription. The `2026-09-17_rf_detr_detector`
benchmark pipeline pairs it with `SimpleHTRModel` for end-to-end HTR.

## Best model

`experiments/experiment1/baseline/train_20260916_115415/checkpoint_best_total.pth`
— exported to ONNX and uploaded to
[PellelNitram/xournalpp-htr-word-detector-rf-detr](https://huggingface.co/PellelNitram/xournalpp-htr-word-detector-rf-detr).

**This is not the detector the project uses.** It loses to WordDetectorYOLO on
recall by a wide margin on the benchmark dataset (66.8% vs 80.1%), so
`2026-09-02_yolo_detector` remains the pipeline of record. The RF-DETR
checkpoint is published so the `2026-09-17_rf_detr_detector` pipeline is
reproducible, not because it is recommended.

Benchmark results (`2026-09-17_rf_detr_detector` pipeline, ONNX inference,
211 ground-truth words), with the YOLO detector alongside for comparison:

| Metric | RF-DETR | WordDetectorYOLO |
|---|---|---|
| Recall | 66.8% | **80.1%** |
| Precision | 77.9% | 73.8% |
| CER (case-sensitive) | **37.1%** | 34.9% |
| CER (case-insensitive) | 36.4% | **34.4%** |
| Recall × (1 − CER_ci) | 42.5% | **52.5%** |
| Word accuracy | **41.8%** | 39.1% |
| Predicted words | 181 | 229 |
| Matched | 141 | 169 |

## Experiments

<!-- Add new experiments below, newest first. -->

### 2026-09-17 — ONNX export, inference class and benchmark

- **Hypothesis:** the IAM validation scores (96.2% recall) carry over to the
  Xournal++ benchmark dataset, beating WordDetectorYOLO on recall.
- **Setup:** `checkpoint_best_total.pth` from the 2026-09-16 baseline run,
  exported at 1024px with `fp16=False`; new `RFDETRWordDetectorModel`
  (onnxruntime only) and `2026-09-17_rf_detr_detector` pipeline.
- **Command:** `uv run python scripts/run_benchmark.py -p 2026-09-17_rf_detr_detector`
- **Results:** **the hypothesis is wrong.** Recall collapses from 96.2% on the
  IAM validation split to 66.8% on the benchmark — the worst of all four
  pipelines, below even the 2024 baseline. The model predicts 181 boxes
  against 211 ground-truth words, so it under-detects rather than
  mis-localising. On the words it does find it is the best of the four
  (word accuracy 41.8%, lowest detector CER), so its boxes crop well for
  SimpleHTR; there are simply too few of them.
- **Validation:** the ONNX decode was checked against the PyTorch checkpoint on
  five validation images — identical box counts, mean IoU 0.998, worst 0.967.
  The YOLO row of the same benchmark sweep reproduced its documented figures
  exactly, so the harness is sound and the gap is real.
- **Conclusion:** a domain gap between IAM forms and rendered Xournal++
  documents, not a training or export defect. Both detectors saw the same IAM
  data, so the transformer appears to generalise off-distribution less well
  here than the convolutional detector. The cheapest untried remedy is a
  confidence-threshold sweep (0.5 is conservative and needs no retraining);
  after that, training on rendered data (issue #150).

### 2026-09-16 — Baseline and learning-rate sweep

- **Hypothesis:** RF-DETR fine-tuned on IAM is a viable word detector, and the
  learning rate is worth tuning.
- **Setup:** RFDETRMedium @ 1024px, IAM-DB from HF Hub converted to COCO,
  85/15 split (seed 42, 1309 train / 230 val forms, ~75 words per image),
  batch 8 × grad-accum 2, 50 epochs, on an L4. Four runs: baseline plus
  lr ∈ {5e-5, 1e-4, 2e-4}. ~2.5 h each.
- **Command:** `bash run_training.sh`
- **Results** (best-recall epoch, IAM validation split):

  | Run | LR | Best ep. | Recall | Precision | mAP@50 | mAP@50:95 |
  |---|---|---|---|---|---|---|
  | experiment1/baseline | 1e-4 | 33 | 96.19% | 96.84% | 97.04% | 89.47% |
  | experiment2/lr5e-5 | 5e-5 | 35 | 96.20% | 96.63% | 96.92% | 89.19% |
  | experiment2/lr1e-4 | 1e-4 | 43 | 96.45% | 96.79% | 97.15% | 89.92% |
  | experiment2/lr2e-4 | 2e-4 | 44 | 96.39% | 96.80% | 97.09% | 89.95% |

- **The sweep is inconclusive.** `config.py` already defaults to lr=1e-4, so
  `experiment2/lr1e-4` re-ran the baseline configuration; with `seed=None` the
  two are independent replicates. They are also the two extremes of the table,
  so run-to-run variance is at least as large as any learning-rate effect.
  Treat lr=1e-4 topping the table as noise, not a finding.
- **Best model:** `experiments/experiment1/baseline/train_20260916_115415/checkpoint_best_total.pth`
  — recall 96.19% on the IAM validation split. Selected over the marginally
  higher `lr1e-4` run because they are the same configuration and `baseline`
  is the honest provenance.
- **Conclusion:** recall peaks mid-training (epochs 33–44) and drifts down
  while precision rises, so late epochs trade away the metric we care about.
  `best_model_metric='map'` therefore does not select the best-recall epoch.
  Learning rate is not a useful knob in [5e-5, 2e-4].

## Current status

Fully implemented and benchmarked: training, ONNX export, HF Hub upload, demo,
and lean ONNX inference via `RFDETRWordDetectorModel` (no `rfdetr` dependency).
The `2026-09-17_rf_detr_detector` benchmark pipeline pairs it with
`SimpleHTRModel`.

**The model is not in production use** — WordDetectorYOLO beats it on recall on
the benchmark dataset, and the project stays on `2026-09-02_yolo_detector`.

Notes for anyone picking this up:

- Verified against **rfdetr 1.10.1**. That release dropped the `base` variant
  for nano/small/medium/large, requires the resolution to be a multiple of
  `patch_size * num_windows` (32, not the 56 older docs mention), and keeps its
  training and ONNX dependencies behind the `train` and `onnx` extras.
- rfdetr 1.10.1 is built on PyTorch Lightning and writes its own TensorBoard
  logs via `tensorboard=True`. Its `model.callbacks` dict still exists but is
  **never invoked**, so a hook registered there fails silently rather than
  erroring.
- The exported classifier has two classes, but `word` is index **0**: RF-DETR
  does not carry the Roboflow COCO category ids through to the model, and index
  1 is never trained (it peaks at a 0.008 score). Getting this wrong makes
  `detect()` return nothing at all.
- `export.py` forces `fp16=False`; rfdetr defaults to fp16, which onnxruntime
  handles poorly on CPU.
- The IAM XML parsing in `train.py` is duplicated from the YOLO detector's
  `train.py`; it is a candidate for `xournalpp_htr/training/shared/`.
- `uv.lock` has not been updated for the `training-word-detector-rf-detr`
  extra, so a fresh clone will re-resolve it.

## Outlook

- Confidence-threshold sweep — 0.5 is conservative and costs nothing to change,
  so this is the cheapest shot at recovering the missing recall.
- Train on rendered Xournal++ data rather than IAM forms alone (issue #150),
  which targets the domain gap directly.
- Set `best_model_metric` to a recall-based metric, since mAP selects a
  checkpoint from the epochs where recall is already declining.
- Try `model.variant=large` and higher `model.resolution`.
- Lift the shared IAM XML parsing into `xournalpp_htr/training/shared/`.
