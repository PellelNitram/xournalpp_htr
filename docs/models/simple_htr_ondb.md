# SimpleHTR-OnDB

[SimpleHTR](simple_htr.md) retrained on IAM-OnDB, the *online* handwriting
database. Its pen strokes are rendered into word images with the same code that
renders Xournal++ pages at inference time, so the training images look like the
images the recogniser sees in production. The best variant trains on those
rendered words **together with** the offline IAM-DB word images.

Same network, preprocessing and decoding as SimpleHTR, only the training data
differs. It is a copy of `xournalpp_htr/training/simple_htr/`, not a shared
module (see [Structure](#structure)). Tracked in issue #150; the original idea
of that issue, training the *detectors* on rendered data, is still open (see
[Outlook](#outlook)). Pipeline contract: [ADR 003](../ADRs/003_define_target_architecture.md),
inference format and training environment:
[ADR 006](../ADRs/006_model_registry_and_training_environment.md), demo:
[ADR 007](../ADRs/007_model_demos_local_only.md).

Source: `xournalpp_htr/training/simple_htr_ondb/`

## Structure

| File | Purpose | Deps |
|------|---------|------|
| `word_splitting.py` | Glue punctuation tokens, cut a line into words at stroke gaps | base |
| `prepare_data.py` | Step 1: render IAM-OnDB lines to word crops + manifest | training |
| `config.py` | Hydra structured config (defaults of SimpleHTR experiment 3) | base |
| `network.py` | `SimpleHTRNet` (CNN+BiLSTM+CTC), copy of SimpleHTR | training |
| `dataset.py` | IAM-DB and rendered-OnDB loaders, combined dataset | training |
| `train.py` | Step 2: training entrypoint (`@hydra.main`) | training |
| `infer.py` | Local torch inference from `.pth` | training |
| `demo.py` | Local Gradio demo (ADR 007) | training |
| `export.py` | ONNX export + HF Hub upload | training |
| `utils.py` | Device, git hash, JSON encoder | training |
| `run_pipeline.sh` | Data preparation command | shell |
| `run_training.sh` | Experiments | shell |

`StrokesDocument` in `xournalpp_htr/documents.py` turns in-memory strokes into a
`Document`, so synthetic strokes go through the unchanged `save_page_as_image`.
`SimpleHTRModel.from_local()` loads an export from a directory, e.g. to
benchmark before publishing.

## GPU training setup

### 1. Clone and install base

```bash
git clone https://github.com/PellelNitram/xournalpp_htr.git
cd xournalpp_htr
bash INSTALL_LINUX.sh
```

### 2. Install the training extra with CUDA PyTorch

```bash
uv sync --extra training-simple-htr-ondb
```

`run_pipeline.sh` and `run_training.sh` use `uv run --extra
training-simple-htr-ondb`, so they also work without this step.

### 3. Verify GPU

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Authenticate with HuggingFace

```bash
uv run huggingface-cli login
```

The IAM-OnDB (`PellelNitram/xournalpp_htr_IAM_OnDB`) and IAM-DB
(`PellelNitram/xournalpp_htr_IAM_DB`) datasets are private and are downloaded on
first use.

### 5. Prepare the data

```bash
bash xournalpp_htr/training/simple_htr_ondb/run_pipeline.sh
```

Writes `data/` next to the script (gitignored); it takes a few minutes. For every
IAM-OnDB line it:

1. glues standalone punctuation tokens onto the neighbouring word (`Mr .` →
   `Mr.`) and rejects lines with ambiguous punctuation (standalone quotes,
   dashes);
2. cuts the strokes into words at the largest horizontal gaps. The transcription
   gives the word count; the line is only kept if exactly that many gaps are at
   least 1.5× larger than the largest remaining one;
3. scales the line so its median word height is drawn log-uniformly from
   10–46 pt, the 5th–95th percentile of the benchmark's ground-truth words
   (median 21.5 pt). The benchmark only supplies this size statistic;
4. renders it with `StrokesDocument.save_page_as_image` at 150 DPI to a JPEG
   and reads it back as grayscale, exactly like `compute_predictions`;
5. crops every word at its stroke box plus 4 px.

The train/val split is by writer (10% of writers in val). Outputs:
`images/<split>/*.png`, `manifest.jsonl`, `stats.json`, and two contact sheets,
`preview_crops.png` and `preview_lines.png`. Look at the previews before
training.

Result on the current dataset: 7,901 of 12,187 lines usable (65%), giving 35,221
train words (195 writers) and 4,214 val words (22 writers). Rejections:
3,666 `ambiguous_gaps`, 599 `ambiguous_punctuation`, 21 other.

### 6. Train

```bash
bash xournalpp_htr/training/simple_htr_ondb/run_training.sh
```

Runs the experiments listed in the script into `experiments/<experiment>/<run>`.
Training sources are switched with `data.use_ondb` and `data.use_iam`; the
default is both. Validation is the union of both val splits; TensorBoard also
gets `cer/val_ondb` and `cer/val_iam`, and checkpoints are selected on the
combined val CER. The IAM-DB part is split 95/5 by form (seed 42).

### 7. Inspect the best model

TensorBoard, on the VM through an SSH tunnel:

```bash
ssh -L 6006:localhost:6006 <vm>
cd xournalpp_htr
uv run --extra training-simple-htr-ondb tensorboard \
    --logdir xournalpp_htr/training/simple_htr_ondb/experiments --port 6006
```

Local Gradio demo:

```bash
uv run --extra training-simple-htr-ondb python -m \
    xournalpp_htr.training.simple_htr_ondb.demo --help
```

### 8. Export to ONNX

```bash
uv run --extra training-simple-htr-ondb python -m \
    xournalpp_htr.training.simple_htr_ondb.export \
    --checkpoint <run>/best_model.pth --output-dir <run>/export
```

### 9. Validate the export, then upload

`tests/test_simple_htr_ondb_export.py` exports a randomly initialised network and
checks that `SimpleHTRModel.from_local` reproduces the PyTorch log-probabilities
(`atol=1e-4`). Run the same comparison on real crops for the checkpoint you are
about to publish (the published one: 200 validation crops, maximum
|log-probability difference| 4.2e-5, identical greedy text for all 200), then
upload:

```bash
uv run --extra training-simple-htr-ondb python -m \
    xournalpp_htr.training.simple_htr_ondb.export \
    --checkpoint <run>/best_model.pth --output-dir <run>/export --upload
```

The upload sends `model.onnx` and `config.json` to
`PellelNitram/xournalpp-htr-simple-htr-ondb`. The word list for beam decoding is
not duplicated there; it is the one published next to SimpleHTR.

## Inference

```python
from xournalpp_htr.inference_models import SimpleHTROnDBModel

model = SimpleHTROnDBModel.from_pretrained()
text = model.recognize(word_image_grayscale)
```

It is a `SimpleHTRModel` subclass bound to the OnDB repository, so the
lexicon-biased beam search works the same way
(`model.use_lexicon(SimpleHTROnDBModel.load_vocabulary())`, then
`recognize(..., decoder="beam")`).

Two pipelines use it, both with the YOLO detector of
`2026-09-02_yolo_detector`:

- `2026-10-17_yolo_detector_simple_htr_ondb`: greedy decoding.
- `2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab`: beam search snapped to
  the nearest known word.

```bash
uv run python scripts/run_benchmark.py -p 2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab
```

## Best model

Experiment 2 (IAM-DB + rendered IAM-OnDB), `experiments/experiment2/combined/best_model.pth`,
epoch 90 — validation CER 6.35%, word accuracy 83.0% (OnDB val 5.9% / 82.5%,
IAM val 6.8% / 83.5%). Published as
[PellelNitram/xournalpp-htr-simple-htr-ondb](https://huggingface.co/PellelNitram/xournalpp-htr-simple-htr-ondb).

Benchmark (211 ground-truth words, same YOLO detector throughout, so precision
73.8% and recall 80.1% for all rows):

| Recogniser | CER (case-ins.) | CER (case-sens.) | R×(1−CER) | Word accuracy |
|---|---|---|---|---|
| IAM, greedy (`2026-09-02_yolo_detector`) | 34.4% | 34.9% | 52.5% | 39.1% (66/169) |
| IAM+OnDB, greedy | 33.2% | 34.3% | 53.5% | 43.2% (73/169) |
| IAM, beam + vocab (`2026-10-05_yolo_detector_beam_vocab`) | 31.7% | 32.3% | 54.7% | 46.7% (79/169) |
| **IAM+OnDB, beam + vocab** | **30.4%** | **31.7%** | **55.7%** | **52.1% (88/169)** |

**This is not the default model.** It is better on every metric, but the
difference is within what 169 matched words can resolve: with beam search the
word accuracy gain is +5.3 points (95% bootstrap interval −0.6 to +11.8,
McNemar p = 0.14) and the CER gain 1.3 points (interval −2.3 to +4.8). The
default pipeline stays `2026-10-05_yolo_detector_beam_vocab` until the result is
replicated (see [Outlook](#outlook)). The pipelines are permanent, so the
comparison can be repeated at any time.

## Experiments

<!-- Add new experiments below, newest first. -->

### 2026-10-22 — IAM-DB plus rendered IAM-OnDB (experiment 2)

- **Hypothesis:** the OnDB-only model lost to the IAM-only model (experiment 1),
  but it saw ~35k words against IAM's ~92k. Training on both adds the stroke
  based, writer-diverse data without giving up IAM, and should beat IAM alone.
- **Setup:** from scratch, same hyperparameters as experiment 1 and SimpleHTR
  experiment 3 (dropout 0.5, augmentation on, lr 0.0005, batch size 64, patience
  25). Train: 35,221 OnDB + 91,658 IAM words; val: 4,214 OnDB + 4,796 IAM words;
  79 characters. Code revision `45449ca` (recorded by the run), one L4 GPU.
- **Command:** `bash xournalpp_htr/training/simple_htr_ondb/run_training.sh`
  (`experiment2`); benchmark with
  `scripts/run_benchmark.py -p 2026-10-17_yolo_detector_simple_htr_ondb[_beam_vocab]`.
- **Results:** ~52 s per epoch, early stopping at epoch 115 (best: 90), 96
  minutes of training. Validation CER 6.35%, word accuracy 83.0%. Benchmark: see
  [Best model](#best-model). Paired against the IAM-only model on the 169 matched
  words with beam search: 19 words only the new model gets right, 10 only the old
  one (McNemar p = 0.14); greedy: 19 vs 12 (p = 0.28). The CER gain is 1.3 points
  in both cases, with a bootstrap interval that includes zero.
- **Best model:** `experiments/experiment2/combined/best_model.pth` — benchmark
  CER 30.4%, word accuracy 52.1% with beam + vocab. The overall best of this
  model, but not significantly better than the default.
- **Conclusion:** mixing the sources helps where OnDB alone hurt, and the
  validation CER on OnDB words also improved (7.2% → 5.9%) over experiment 1.
  The benchmark is too small to confirm the gain. Replicate with other seeds and
  enlarge the benchmark before changing the default.

### 2026-10-21 — Crop analysis of the OnDB-only model

- **Hypothesis:** the OnDB-only model's drop from 77.6% word accuracy on rendered
  OnDB words to 40.2% on the benchmark comes from crop mismatch (training crops
  vs. YOLO crops) or from stroke colour.
- **Setup:** `scripts/run_benchmark.py --crop-analysis` for both pipelines,
  compared per word with the ground truth crops and the training crops.
- **Results:**
    - The YOLO crops of the 169 matched words are essentially the ground truth
      boxes (ink width ratio 1.00 at the median; 7 words truncated, 3 clipped in
      height). The only systematic difference is margin: ~3 px of white in
      training vs. 0–1 px from YOLO (ink fills 89% vs. 97% of the crop height).
    - The benchmark has black (850 strokes) and blue `#3333cc` strokes (229, two
      documents; mid-grey after grayscale conversion), training is black only.
      The gap is the same size on both: IAM vs. OnDB-only 48% vs. 41% word
      accuracy on black documents, 43% vs. 36% on blue ones.
    - The loss is concentrated in a few documents; the largest one (94 words) is
      identical for both models. Per word: 56 right in both, 78 wrong in both,
      23 only IAM, 12 only OnDB (McNemar p = 0.09).
    - Much of the benchmark is not newspaper English: digits, symbols,
      German, all-caps words, technical terms (`Relu`, `overfitting`), and some
      detections merge two words.
- **Best model:** n/a.
- **Conclusion:** crop geometry and ink colour do not explain the gap; the
  difference is in the handwriting and content. Vocabulary snapping also breaks
  correct non-dictionary tokens (`fr`, `std`, `ein`, `Relu`), for both models.

### 2026-10-17 — Rendered IAM-OnDB only (experiment 1)

- **Hypothesis:** a SimpleHTR trained on the Xournal++-rendered strokes of
  IAM-OnDB generalises better to Xournal++ documents than one trained on scanned
  IAM forms, because the training images come from the inference rendering.
- **Setup:** from scratch with SimpleHTR experiment 3's settings. Train: 35,221
  words (195 writers); val: 4,214 words (22 unseen writers); 76 characters.
  Code revision `ddc9e88` (recorded by the run), one L4 GPU.
- **Command:** `bash xournalpp_htr/training/simple_htr_ondb/run_training.sh`
  (`experiment1`); benchmark as above.
- **Results:** ~16.5 s per epoch, early stopping at epoch 139 (best: 114), 34
  minutes. Validation CER 7.2%, word accuracy 77.6% — but on the benchmark:

| Recogniser | CER (case-ins.) | R×(1−CER) | Word accuracy |
|---|---|---|---|
| IAM, greedy | 34.4% | 52.5% | 39.1% (66/169) |
| OnDB only, greedy | 39.7% | 48.3% | 36.7% (62/169) |
| IAM, beam + vocab | 31.7% | 54.7% | 46.7% (79/169) |
| OnDB only, beam + vocab | 39.7% | 48.3% | 40.2% (68/169) |

  The identical CER of the two OnDB rows is a coincidence: both total 325 edit
  characters, because beam search fixes some words and breaks others by the
  same amount (checked word by word).
- **Best model:** `experiments/experiment1/baseline/best_model.pth` — benchmark
  CER 39.7%. Not published; worse than the IAM-only model.
- **Conclusion:** worse than IAM alone on every metric, though the word accuracy
  gap (−6.5 points with beam search, McNemar p = 0.09) is borderline. See the
  crop analysis above for why geometry and colour are not the cause.

## Current status

- [x] Strokes → rendered word crops through the inference rendering
      (`StrokesDocument`, `prepare_data.py`), writer-based split
- [x] Training on OnDB, IAM-DB or both, with per-source validation
- [x] ONNX export, HF Hub upload, `SimpleHTROnDBModel`
- [x] Pipelines `2026-10-17_yolo_detector_simple_htr_ondb` and
      `2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab`
- [x] Local Gradio demo (copied from SimpleHTR, not exercised for this model)

Known issues:

- **The benchmark is too small.** 169 matched words, a standard error of about
  3.8 points on word accuracy; the combined model's gain is not significant.
- **Selection bias in the data.** Only the 65% of lines with clear word gaps are
  used, which skews towards neat, widely spaced writing. Lines with ambiguous
  punctuation are dropped too.
- **Label noise.** The transcription is the corrected text, so crossed-out words
  and stray marks carry a label that doesn't match the ink.
- **Single run.** All results are one seed; run-to-run variance is unmeasured.
- Stroke width cannot be randomised: `save_page_as_image` draws every stroke
  with the same line width, as it does for real documents.
- `demo.py` is not exercised, and the ONNX notebook of the other models doesn't
  exist for this one; validation is the offline test plus a manual check.

## Outlook

- **Replicate with other seeds.** Train experiment 2 several times to separate
  training noise from the data effect (about 1.5 h per run on an L4).
- **Grow the benchmark.** More annotated pages are the real fix; 169 words
  cannot resolve a 1–2 point CER difference. Report paired statistics
  (McNemar, bootstrap) with every comparison.
- **Cheap data tweaks.** Match the 0–1 px margin of the YOLO crops instead of
  4 px, and add the blue pen colour to the rendering (`StrokesDocument` already
  takes a colour). Test them as their own experiments, separately from the
  data mix.
- **Fine-tune instead of mixing.** Start from the IAM-only checkpoint and
  fine-tune on OnDB.
- **Mixing ratio.** IAM is 72% of the combined training set; try upweighting
  OnDB or oversampling the writers it adds.
- **Keep more lines.** The gap rule drops 30% of lines. Aligning the transcription
  to the strokes with the current model would keep more, at the cost of biasing
  the data towards what the model already reads.
- **Gate the vocabulary snapping** on the recogniser's confidence, so
  correct non-dictionary tokens (`fr`, `std`, `Relu`) are left alone.
- **Detectors (the original scope of #150).** Train the YOLO and RF-DETR
  detectors on rendered OnDB *pages* (`PageDatasetFromOnline` already places
  words on pages with their boxes). RF-DETR's gap between 96.2% recall on IAM
  validation and 66.8% on the benchmark makes it the sensitive test case.
