---
library_name: onnx
license: mit
tags:
  - handwriting-recognition
  - htr
  - online-handwriting
  - ctc
  - xournalpp
---

# xournalpp-htr-online-htr

Stroke-based handwriting recogniser for
[Xournal++ HTR](https://github.com/PellelNitram/xournalpp_htr): a 3-layer
bidirectional LSTM (128 units) with CTC loss, after
[Carbune et al. (2020)](https://doi.org/10.1007/s10032-020-00350-4) and ported
from [OnlineHTR](https://github.com/PellelNitram/OnlineHTR). It reads the pen
strokes of one word or line directly, with no rendered image. Recognition only;
a word detector has to group the strokes first.

- **Code and training:** <https://github.com/PellelNitram/xournalpp_htr>
  (`xournalpp_htr/training/online_htr/`)
- **Documentation:** `docs/models/online_htr.md` has the experiments and the
  error analysis.

## Files

- `model.onnx`: input `ink` `(T, N, 3)`, output `log_probs` `(T, N, classes)`.
- `config.json`: charset (the CTC blank is the last class) and
  `features.points_per_unit`.

## Usage

```python
from xournalpp_htr.inference_models import OnlineHTRModel

model = OnlineHTRModel.from_pretrained()
text = model.recognize(strokes)  # [(x, y), ...] for one word -> str
```

Strokes are `(x, y)` arrays in writing order, in document units with y pointing
down. Features per point are `(dx, dy, n)` after scaling to unit ink height and
resampling by arc length; `n` marks the start of a stroke. There is no time
channel, as Xournal++ stores no timestamps. Optional beam search snapped to a
word list: `model.use_lexicon(words)`, then `recognize(strokes, decoder="beam")`.

## Training

IAM-OnDB lines and words (words cut from the lines at stroke gaps), 80
characters, writer-based split with 10% of the writers held out, augmentation
by x-scaling, shear and rotation. Best validation CER 5.2% (epoch 30). No
Xournal++ benchmark data is used in training.

## Results

Benchmark of 7 handwritten Xournal++ documents by 7 writers (406 words), with
the online word detector grouping the strokes (precision 84.1%, recall 87.4%):

| Decoding | CER (case-ins.) | Recall × (1 − CER) | Exact recall |
|---|---|---|---|
| Greedy | 28.5% | 62.5% | 38.4% |
| Beam + word list | 27.1% | 63.8% | 45.8% |

Exact recall is the share of all words found and read exactly, ignoring case.

## Limitations

- Out-of-domain: trained on IAM-OnDB, not on Xournal++ notes.
- Punctuation and symbols are read poorly (about 10% of such words correct even
  with perfect grouping); words outside the word list are almost never read
  correctly with beam search.
- The benchmark is small (406 words), so differences of a few points are not
  reliable.

## License and data

The model weights and code are released under the MIT license. The model was
trained on IAM-OnDB, which has its own terms of use.
