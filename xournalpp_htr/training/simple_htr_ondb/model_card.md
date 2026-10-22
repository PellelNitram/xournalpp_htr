---
library_name: onnx
pipeline_tag: image-to-text
tags:
  - handwriting-recognition
  - htr
  - ctc
  - xournalpp
---

# xournalpp-htr-simple-htr-ondb

Word-level handwritten text recognition for
[Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr): the
[SimpleHTR](https://github.com/githubharald/SimpleHTR) network (5 CNN layers,
2 bidirectional LSTM layers, CTC) trained on **IAM-DB word images plus IAM-OnDB
strokes rendered like Xournal++ pages**. It reads one word image and returns its
text.

- **Code and training:** <https://github.com/PellelNitram/xournalpp_htr>
  (`xournalpp_htr/training/simple_htr_ondb/`)
- **Documentation:** [`docs/models/simple_htr_ondb.md`](https://github.com/PellelNitram/xournalpp_htr/blob/master/docs/models/simple_htr_ondb.md)
  has the data preparation, all experiments and the open questions.

## Files

- `model.onnx`: the network, input `(batch, 1, 32, 128)` float, output CTC
  log-probabilities `(time, batch, classes)`.
- `config.json`: charset (79 characters, CTC blank is the last class), input
  size and normalisation.

## Usage

```python
from xournalpp_htr.inference_models import SimpleHTROnDBModel

model = SimpleHTROnDBModel.from_pretrained()
text = model.recognize(word_image_grayscale)  # uint8, any size
```

Decoding is greedy by default. `model.use_lexicon(SimpleHTROnDBModel.load_vocabulary())`
followed by `recognize(..., decoder="beam")` snaps the result to the nearest known
word; the word list is the one published with
[xournalpp-htr-simple-htr](https://huggingface.co/PellelNitram/xournalpp-htr-simple-htr).

In the repository the model runs in two benchmark pipelines, both with the YOLO
word detector:
`2026-10-17_yolo_detector_simple_htr_ondb` and
`2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab`.

## Training data

| Source | Train words | Val words |
|---|---|---|
| IAM-DB (offline, scanned forms), split 95/5 by form | 91,658 | 4,796 |
| IAM-OnDB (online strokes), rendered at 150 DPI with the Xournal++ page renderer, split by writer | 35,221 | 4,214 |

Trained from scratch: dropout 0.5, augmentation, learning rate 0.0005, batch size
64, early stopping. Validation CER 6.35%, word accuracy 83.0%.

## Benchmark

On the Xournal++ HTR benchmark (211 ground-truth words, YOLO detector, so
detection is the same for every row), compared with the IAM-only SimpleHTR:

| Recogniser | CER (case-ins.) | Word accuracy |
|---|---|---|
| SimpleHTR (IAM), greedy | 34.4% | 39.1% |
| This model, greedy | 33.2% | 43.2% |
| SimpleHTR (IAM), beam + vocabulary | 31.7% | 46.7% |
| **This model, beam + vocabulary** | **30.4%** | **52.1%** |

The improvement is **within noise** on 169 matched words (word accuracy +5.3
points, 95% interval −0.6 to +11.8). This model is therefore not the default of
Xournal++ HTR; the experiment still needs replication.

## Limitations

- Recognises single words. Text lines have to be split into words first.
- Trained on English text from IAM; the benchmark also contains German words,
  digits and symbols, which it handles poorly.
- Trained on lines whose words could be separated cleanly (65% of IAM-OnDB), so
  it may be biased towards neat, widely spaced writing.
- The training databases come with their own terms of use; check them before
  any use beyond research.
