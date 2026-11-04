# WordDetectorOnline model

Stroke-based word detector. It works on the pen strokes of a page instead of a
rendering of it: a small gradient boosted model scores each pair of consecutive
strokes as "same word" or not, and the stroke sequence is cut into words where
the score falls below a threshold. It performs detection only; a recognizer
([SimpleHTR](simple_htr.md) or [SimpleHTR-OnDB](simple_htr_ondb.md)) reads the
words afterwards. Integrated into Xournal++ HTR according to the
[ADRs](../ADRs/) (in particular
[ADR 006](../ADRs/006_model_registry_and_training_environment.md)), with one
exception: the artifact is a numpy file, not ONNX (see below). Tracked in issue
#168.

The source lives under
[`xournalpp_htr/training/word_detector_online/`](https://github.com/PellelNitram/xournalpp_htr/tree/master/xournalpp_htr/training/word_detector_online).

## Structure (ADR 006)

| File | Purpose | Deps |
| --- | --- | --- |
| `strokes.py` | `StrokePage`, the strokes of a page plus word labels | base |
| `features.py` | Spatial features of consecutive strokes | base |
| `model.py` | Boosted trees as plain arrays, numpy evaluation, cutting into words | base |
| `detectors.py` | The boosted detector and the tuning of its merge threshold | base |
| `iam_ondb.py` | IAM-OnDB forms as training pages, with the gaps between words shrunk | `training-simple-htr-ondb` |
| `data.py` | The benchmark documents as labelled stroke pages | base |
| `metrics.py`, `evaluate.py` | Exact and box-coverage scoring, evaluation CLI | base |
| `export.py` | Trains on IAM-OnDB, writes `model.npz` + `config.json`, HF Hub upload | `training-simple-htr-ondb` |
| `model_card.md` | Published as the HF repo's `README.md` | none |

The HF-Hub-backed inference class lives in `xournalpp_htr/inference_models.py` as
`OnlineWordDetectorModel`. Deviations from the model conventions: there is no
ONNX file (the trees are stored as plain arrays and evaluated in numpy, so
inference needs neither onnxruntime nor scikit-learn), no separate checkpoint
(training takes minutes, `export.py` trains and exports in one go), no Hydra
config and no Gradio demo yet.

## Training setup (step-by-step)

No GPU is needed. Training takes minutes on a CPU, most of it spent reading
IAM-OnDB (not timed separately).

1. Clone and install the base package (`bash INSTALL_LINUX.sh`), then
   `uv sync --extra training-simple-htr-ondb` for the IAM-OnDB loader.
2. Authenticate with HuggingFace (`hf auth login`): the IAM-OnDB dataset repo
   is private.
3. Check the install with `make tests-not-slow`.
4. Score a detector on the benchmark, leaving the training data out of it:

   ```bash
   uv run python -m xournalpp_htr.training.word_detector_online.evaluate
   ```

   It trains on IAM-OnDB pages and scores every benchmark document.
   `--iam-seed` changes the draw of IAM-OnDB forms and `--gap-squeeze LOW HIGH`
   the shrinking of the gaps between words.
5. Export:

   ```bash
   uv run python -m xournalpp_htr.training.word_detector_online.export \
       --output-dir exports/word_detector_online
   ```

6. Validate: the test `tests/test_word_detector_online_model.py` checks that the
   numpy evaluation reproduces scikit-learn's probabilities, and benchmarking the
   pipeline that loads the model must give the numbers under *Best model*.
7. Upload with `export.py --upload` once validated.

## Inference

```python
from xournalpp_htr.inference_models import OnlineWordDetectorModel

model = OnlineWordDetectorModel.from_pretrained()
words = model.group_strokes(strokes)  # [(x, y), ...] -> [frozenset({0, 1}), ...]
```

`strokes` is a list of `(x, y)` arrays in writing order and document units. The
benchmark pipeline `2026-11-03_online_word_detector_hub_simple_htr_ondb` uses the
published model, renders each word from its own strokes and reads it with
SimpleHTR-OnDB.

## Best model

Detector trained on 700 IAM-OnDB forms stitched into pages with the gaps between
words shrunk by a factor of 0.3 to 1.0, merge threshold tuned on out-of-fold
probabilities, no post-processing. Benchmark pipeline
`2026-11-03_online_word_detector_hub_simple_htr_ondb`
(SimpleHTR-OnDB reading crops rendered from each word's strokes): precision
84.1%, recall 87.4%, CER 27.3% (case-insensitive), recall × (1 − CER) 63.6%.
YOLO with SimpleHTR scores 77.0%, 72.4%, 27.5% and 52.5% on the same 406 words;
a paired bootstrap puts the online pipeline's lead in the combined score at +11
points (95% interval [+4.5, +15.9] resampling documents), driven by recall.

## Experiments

The pipelines and detectors of the experiments below, and the code only they
used, were removed because users cannot run them: they refit a detector on every
run or need the private IAM-OnDB dataset. The experiments stay documented here;
the code is in the git history up to commit `1e4bcb2`, for example
`git show 1e4bcb2:xournalpp_htr/training/word_detector_online/postprocess.py`.
The pipeline names below are the ones that existed then.

All numbers are on the benchmark of 7 documents by 7 writers (406 words) with
the benchmark's box-coverage matching. The benchmark is never part of the
training data of the IAM-OnDB detectors; the benchmark-trained detectors are
scored leave-one-document-out, which also holds out the writer.

### 2026-11-02 — Stability and post-processing

- **Hypothesis:** the squeeze range and the draw of IAM-OnDB forms hardly matter,
  and merging fragments and dropping underlines improves the combined score.
- **Setup:** detection scores of the IAM-trained detector for squeeze ranges
  0.5–1.0, 0.2–1.0, 0.3–1.3 and 0.5–1.3, for five seeds of the form draw, and
  with post-processing tuned on 300 other IAM-OnDB forms (`postprocess.py`).
- **Results:** all squeezed ranges give detection F1 0.85–0.86 (unsqueezed 0.84).
  Five form draws give F1 0.86 each (precision 85–88%, recall 85–87%).
  Post-processing raises precision from 84.1% to 87.1% and lowers recall from
  87.4% to 86.2%, so the combined score falls from 63.6% to 62.8%.
- **Conclusion:** keep the 0.3–1.0 squeeze and no post-processing; the detector
  is stable across draws of the training forms.

### 2026-11-02 — Squeezing the gaps between words

- **Hypothesis:** the detector merges neighbouring words because IAM-OnDB word
  gaps are wider than the benchmark's.
- **Setup:** per IAM-OnDB line, the gaps between words shrink by a random factor
  between 0.3 and 1.0 (`iam_ondb.py`).
- **Results:** recall 81.5% → 87.4%, precision 88.5% → 84.1%, CER 31.9% → 27.3%,
  recall × (1 − CER) 55.5% → 63.6%.
- **Conclusion:** adopted. Remaining errors: hard handwriting, fragmented words
  and stray strokes (underlines, arrows) attached to words.

### 2026-11-01 — Crops rendered from each word's own strokes

- **Hypothesis:** crops cut from the page render contain neighbouring words,
  which hurts recognition.
- **Setup:** each crop is rendered from the strokes of its word alone.
- **Results:** with SimpleHTR-OnDB CER 36.6% → 31.9% and recall × (1 − CER)
  51.7% → 55.5%; with SimpleHTR 38.4% → 34.2%.
- **Conclusion:** adopted; this is an advantage over the vision-based detectors.

### 2026-11-01 — Training on IAM-OnDB pages

- **Hypothesis:** IAM-OnDB, with many writers, can replace the benchmark
  documents as training data.
- **Setup:** lines of a form are combined at their original positions into pages
  (line breaks included); words come from gap cutting
  (`simple_htr_ondb/word_splitting.py`); optional synthetic underlines, arrows,
  boxes and connector curves.
- **Results:** single IAM lines: detection F1 0.65; stitched pages: 0.84;
  synthetic non-text strokes: 0.83, no gain. Reference, benchmark-trained:
  0.77.
- **Conclusion:** stitched pages are what matters. The word labels follow a gap
  rule and only lines where it works cleanly are kept (about 1,800 of 5,000
  examined lines were rejected, mostly as ambiguous gaps), which is a known
  limitation.

### 2026-10-30 — Detectors trained on the benchmark

- **Hypothesis:** an algorithm on the strokes can find words as well as the
  vision-based detectors.
- **Setup:** gap threshold, single-linkage clustering and boosting on features
  of consecutive strokes, leave-one-document-out on the benchmark.
- **Results:** detection F1 (box coverage) 0.77 for the gap threshold, 0.65 for
  clustering and 0.76 for boosting with threshold 0.5, 0.77 with the threshold
  tuned on held-out documents; the tuned boosting pipeline reaches recall 81.3%
  at precision 73.8%.
- **Conclusion:** boosting works best; the benchmark alone is too small to train
  on honestly, which led to IAM-OnDB.

## Current status

- `OnlineWordDetectorModel` loads `model.npz` + `config.json` and groups strokes.
- One benchmark pipeline, `2026-11-03_online_word_detector_hub_simple_htr_ondb`,
  uses the published detector; its numbers are under *Best model*.
- Known issues: in a sample of the wrong crops about a quarter were merged
  words, and split words mostly show up as spurious boxes; strokes of arrows and underlines next to words become spurious boxes or steal a
  word's match; recognition errors on hard handwriting are not addressed here.
- The benchmark is small, so differences of a few points are not reliable.
- Missing compared with the model conventions: Gradio demo, Hydra config.

## Outlook

- Better recognition of clean but misread crops (fine-tune the recognizer on
  stroke renders with realistic grouping noise).
- Merge fragments and drop non-word strokes with a learned model instead of
  rules.
- Word labels for IAM-OnDB that do not depend on a gap rule.
- A Gradio demo to inspect the groups visually.
