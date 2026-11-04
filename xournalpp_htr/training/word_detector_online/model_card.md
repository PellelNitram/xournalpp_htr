---
library_name: numpy
license: gpl-2.0
tags:
  - handwriting-recognition
  - htr
  - word-detection
  - online-handwriting
  - xournalpp
---

# xournalpp-htr-word-detector-online

Stroke-based word detector for
[Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr). It works on the
pen strokes of a page, not on a rendering of it: a small gradient boosted model
scores each pair of consecutive strokes as "same word" or not, and the stroke
sequence is cut into words where the score falls below a threshold. It performs
detection only; a recognizer such as SimpleHTR reads each word afterwards.

- **Code and training:** <https://github.com/PellelNitram/xournalpp_htr>
  (`xournalpp_htr/training/word_detector_online/`)
- **Documentation:** `docs/models/word_detector_online.md` has the experiments
  and the open questions.

## Files

- `model.npz`: the 100 boosted regression trees as plain arrays (no pickle).
- `config.json`: merge threshold, feature names and the training setup.

## Usage

```python
from xournalpp_htr.inference_models import OnlineWordDetectorModel

model = OnlineWordDetectorModel.from_pretrained()
words = model.group_strokes(strokes)  # [(x, y), ...] -> [frozenset({0, 1}), ...]
```

Evaluation is plain numpy, so no scikit-learn or onnxruntime is needed. The
strokes are a list of `(x, y)` arrays in writing order and in document units
(points); the result is disjoint sets of stroke indices, one per word.

## Training

Trained on 700 random IAM-OnDB forms stitched into pages, with the gaps between
words shrunk by a random factor between 0.3 and 1.0 per line, so that the model
also learns tight word spacing. No Xournal++ benchmark document is involved in
training. The merge threshold is tuned on out-of-fold probabilities of the
training pages.

## Results

Benchmark of 7 handwritten Xournal++ documents by 7 writers (406 words), with
the detector followed by SimpleHTR-OnDB on crops rendered from each word's
strokes (pipeline `2026-11-03_online_word_detector_hub_simple_htr_ondb`):

| Metric | Value |
|---|---|
| Detection precision | 84.1% |
| Detection recall | 87.4% |
| CER (case-insensitive) | 27.3% |
| Recall × (1 − CER) | 63.6% |

The vision-based YOLO detector with SimpleHTR scores 77.0%, 72.4%, 27.5% and
52.5% on the same words.

## Limitations

- It needs stroke data and cannot read scanned or photographed handwriting.
- Strokes of arrows, underlines and diagrams that sit next to text are
  sometimes merged into a word or reported as a word of their own.
- The benchmark is small (7 documents), so differences of a few points are not
  reliable.

## License and data

- **License:** GPL-2.0, like the code of
  [Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr).
- **Training data:**
  [IAM-OnDB](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
  (Liwicki and Bunke, ICDAR 2005), used under its terms of use. The data is not
  redistributed; the model holds only the parameters of the trees.
- **Evaluation data:** the
  [Xournal++ HTR benchmark](https://huggingface.co/datasets/PellelNitram/xournalpp_htr_benchmark),
  which is not used for training.
