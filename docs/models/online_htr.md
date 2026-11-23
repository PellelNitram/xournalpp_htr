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

`OnlineHTRModel` (`xournalpp_htr/inference_models.py`) runs the ONNX model on
the strokes of one word or line. No page rendering, no crops.

```python
from xournalpp_htr.inference_models import OnlineHTRModel

model = OnlineHTRModel.from_pretrained()      # or .from_local(export_dir)
text = model.recognize(strokes)               # strokes: list of (x, y) arrays

model.use_lexicon(vocabulary)                 # word list, as for SimpleHTR
text = model.recognize(strokes, decoder="beam")
```

The weights are on the HF Hub as
[PellelNitram/xournalpp-htr-online-htr](https://huggingface.co/PellelNitram/xournalpp-htr-online-htr).
To use a local export instead (e.g. a new experiment), set
`ONLINE_HTR_MODEL_DIR=<export dir>` for the pipelines.

Pipelines (both group strokes with `WordDetectorOnline`):

| Pipeline | Decoding |
|---|---|
| `2026-11-22_online_word_detector_online_htr` | greedy |
| `2026-11-22_online_word_detector_online_htr_beam_vocab` | beam search, snapped to the SimpleHTR word list |

## Best model

`experiments/online_htr/final/best_model.pth` (epoch 30 of 55) — validation
CER 5.18%, word accuracy 84.1%, line accuracy 37.9%. Published on the HF Hub
(see [Inference](#inference)); the hub copy reproduces the benchmark numbers below.

Benchmark (406 ground-truth words, same detector throughout, so precision
84.1%, recall 87.4%, 355 matched). Exact recall is the share of *all* words found
and read exactly (case-insensitive), with a 95% bootstrap interval over pages:

| Pipeline | CER (case-ins.) | R×(1−CER) | Word acc. | Exact recall | ≤1 edit recall |
|---|---|---|---|---|---|
| OnlineHTR, greedy | 28.5% | 62.5% | 43.9% (156/355) | 38.4% (33.3–44.6) | 61.6% |
| **OnlineHTR, beam + vocab** | **27.1%** | **63.8%** | 52.4% (186/355) | **45.8%** (41.3–54.5) | **64.0%** |
| SimpleHTR-OnDB, greedy (`2026-11-03_online_word_detector_hub_simple_htr_ondb`) | 27.3% | 63.6% | 53.8% | not computed | – |

OnlineHTR with the word list is level with SimpleHTR-OnDB. The comparison is
not like for like (SimpleHTR-OnDB without its own beam + vocab variant), and
the intervals are wide. The comparison table with the new metrics for all
pipelines is still to be recomputed.

## Experiments

### 2026-11-22 — Per-word error analysis on the benchmark

- **Hypothesis:** the recogniser, not the detector, limits the benchmark
  result.
- **Setup:** `2026-11-22_online_word_detector_online_htr_beam_vocab` with the
  final model. Every ground-truth word is also recognised from its *true*
  strokes ("oracle detection": perfect grouping) and the two outcomes compared.
  Scratch script, not in the repo.
- **Results:**

  | Outcome | Words | Share | Oracle reads it right |
  |---|---|---|---|
  | Exact | 186 | 45.8% | 185 |
  | Found, misread | 169 | 41.6% | 24 |
  | No matching box | 51 | 12.6% | 27 |

  Exact recall with perfect grouping would be 58.1% (beam) or 49.5% (greedy).

  | Slice (oracle, beam) | Words | Oracle exact | Pipeline exact |
  |---|---|---|---|
  | In the word list | 304 | 74.0% | – |
  | Not in the word list | 102 | 10.8% | – |
  | Contains punctuation or symbols | 66 | 10.6% | 4.5% |
  | Length ≤ 3 | 146 | 73.3% | 57.5% |
  | Length ≥ 8 | 67 | 25.4% | 20.9% |
  | 9+ strokes | 60 | 36.7% | 26.7% |

  Ignoring punctuation at the word edges: pipeline 48.8%, oracle 60.3%.
- **Conclusion:** the detector costs about 12 points (merged, split and padded
  groups), the recogniser about 42. Three causes: letter confusions
  (`this`→`ters`, `detection`→`letaon`), words outside the word list (technical
  terms, German, the writers' typos), and punctuation, which the model barely
  reads (greedy on the true strokes gets 14% of such words) and the word-list
  decoder removes. Next: in-domain training data, a fallback to greedy when the
  word list fits badly, and punctuation in the training labels.

### 2026-11-22 — Beam search with a word list

- **Hypothesis:** the word list that gives SimpleHTR its gain helps here too.
- **Setup:** `OnlineHTRModel.use_lexicon(SimpleHTRModel.load_vocabulary())` and
  `shared/ctc_decoding.beam_decode`, final model, same detector.
- **Command:**
  `ONLINE_HTR_MODEL_DIR=experiments/online_htr/final/export uv run --extra training-online-htr python scripts/run_benchmark.py --pipeline 2026-11-22_online_word_detector_online_htr_beam_vocab`
- **Results:** see [Best model](#best-model): +7.4 points exact recall and +8.5
  points word accuracy over greedy, larger than the interval width. Beam search
  without the word list gained little on the validation set.
- **Conclusion:** the word list is what matters. Compare against SimpleHTR-OnDB
  with *its* beam + vocab variant before ranking the two.

### 2026-11-22 — Final training run

- **Hypothesis:** the best settings from the exploration, trained to
  convergence, give a usable recogniser.
- **Setup:** IAM-OnDB lines and words, writer-based split (10% of writers
  validation, seed 42; 10 970 + 35 221 training and 1 216 + 4 214 validation
  line/word samples), no test split (the benchmark pipeline is the test). 3 BiLSTM layers of 128, dropout 0.1, learning rate 0.002,
  batch 64, augmentation on, `epoch_max=60`, `patience_max=25`. Code revision
  `444d0d6`, one RTX 3060, about 110 s per epoch.
- **Command:** `xournalpp_htr.training.online_htr.train` with Hydra overrides
  `model.nodes_per_layer=128 model.dropout=0.1 training.learning_rate=0.002
  training.epoch_max=60 data.data_path=...`.
- **Results:** best validation CER 5.18% at epoch 30 (lines 5.04%, words 5.22%).
  The run crashed at epoch 55 with `cuDNN error: CUDNN_STATUS_NOT_SUPPORTED` in
  the LSTM forward pass, which is also where early stopping would have stopped
  it, so nothing was lost. Cause not investigated; the error message suggests a
  non-contiguous packed input.
- **Best model:** `experiments/online_htr/final/best_model.pth` — CER 5.18%.
  Overall best.
- **Conclusion:** validation CER fell from 6.5% to 5.2% against the exploration
  but the benchmark CER stayed at about 28%. The bottleneck is the gap between
  IAM-OnDB and real Xournal++ notes, not capacity or training time.

### 2026-11-22 — Exploration (15 epochs each)

- **Hypothesis:** which of width, depth, learning rate, dropout, batch size and
  the data level (words only vs lines + words) matter.
- **Setup:** the final run's data and split, `epoch_max=15`, same GPU.
- **Results (best validation CER):**

  | Run | Val CER |
  |---|---|
  | 3×128 (baseline) | ~0.068 |
  | 3×256 | 0.065 (epoch 9) |
  | 3×128, lr 0.002, dropout 0.1 | ~0.066 (ONNX export checked against PyTorch: 0.0650 vs 0.0647) |
  | 4×128 | ~0.068 |
  | words only | 0.137 (batch 128: 0.207, undertrained) |

- **Conclusion:** differences between the first four are small. Words only was
  clearly worse after 15 epochs, but it was also undertrained, so the gain from
  adding lines is not isolated. 3×128 with lr 0.002 and dropout 0.1 was chosen
  over 3×256: nearly the same CER at lower cost.

## Current status

Implemented and tested: training, ONNX export, `OnlineHTRModel`, two benchmark
pipelines, and the new benchmark metrics (exact and within-one-edit recall with
bootstrap intervals). The model is on the HF Hub. Known issues: cuDNN crash unexplained; punctuation and out-of-list
words are read poorly.

## Outlook

- Fine-tune on in-domain Xournal++ handwriting.
- Fall back to greedy decoding when the word list matches badly; punctuation in
  the training labels.
- Local Gradio demo (ADR 007).
- Benchmark all pipelines with the new metrics; SimpleHTR-OnDB with beam + vocab
  as the fair comparison.
