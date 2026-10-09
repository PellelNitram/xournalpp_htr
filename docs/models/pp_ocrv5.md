# PP-OCRv5

PaddlePaddle's pretrained
[PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) text recognition and
detection, tried as an alternative to [SimpleHTR](simple_htr.md) and
[TrOCR](trocr.md) (issue #157). The mobile English recognition model and the
mobile detection model are used as they are, run as ONNX through
[`rapidocr`](https://github.com/RapidAI/RapidOCR).

**Exception to ADR 006:** the weights are not our own export; `rapidocr`
downloads them on first use. The dependency sits behind the optional `ppocr`
extra.

Two pipelines, to answer two questions:

| Pipeline | Question |
|---|---|
| `2026-10-09_yolo_detector_ppocrv5` | Is PP-OCRv5 a better *recogniser* than SimpleHTR? YOLO detector unchanged, so precision and recall are identical to the other YOLO pipelines. |
| `2026-10-09_ppocrv5_det_rec` | Does PP-OCRv5 work as a full pipeline? Its own line detector, split into words by `rapidocr`. |

## GPU training setup

Not applicable: we have not trained or finetuned PP-OCRv5.

## Inference

```bash
uv sync --extra ppocr
```

```python
from xournalpp_htr.inference_models import PPOCRv5Model

model = PPOCRv5Model.from_pretrained()
texts = model.recognize_batch(word_images_grayscale)  # recognition only
words = model.read_page(page_image_grayscale)  # [(text, BoundingBox), ...]
```

Benchmark:

```bash
uv run --extra ppocr python scripts/run_benchmark.py -p 2026-10-09_yolo_detector_ppocrv5
uv run --extra ppocr python scripts/run_benchmark.py -p 2026-10-09_ppocrv5_det_rec
```

## Best model

Neither pipeline replaces the default. The recognizer-only pipeline
`2026-10-09_yolo_detector_ppocrv5` has the lowest CER of all pipelines
(**28.4%** case-insensitive) but a lower word accuracy (**42.0%**) than
SimpleHTR (46.7%) and TrOCR large (48.5%); see
[Experiments](#experiments).

## Experiments

### 2026-10-09 — PP-OCRv5 as recogniser and as full pipeline

- **Hypothesis:** PP-OCRv5 beats SimpleHTR (CER 31.7%, word accuracy 46.7%,
  `2026-10-05_yolo_detector_beam_vocab`) on handwritten word crops, and
  possibly works without our YOLO detector.
- **Setup:** benchmark dataset `latest`; code revision `712027d`; CPU only.
  Mobile English recognition model and mobile detection model of
  PP-OCRv5.
- **Command:** `uv run --extra ppocr python scripts/run_benchmark.py -p <pipeline>`.
- **Results:**

| Pipeline | Precision | Recall | CER (case-insens.) | CER (case-sens.) | R×(1-CER) | Word Acc |
|---|---|---|---|---|---|---|
| `2026-10-05_yolo_detector_beam_vocab` (SimpleHTR, for reference) | 73.8% | 80.1% | 31.7% | n/a | n/a | 46.7% |
| `2026-10-09_yolo_detector_trocr_large_strip_punct` (for reference) | 73.8% | 80.1% | 29.3% | 37.9% | 56.6% | 48.5% (82/169) |
| `2026-10-09_yolo_detector_ppocrv5` (recognizer only) | 73.8% | 80.1% | **28.4%** | 29.4% | **57.3%** | 42.0% (71/169) |
| `2026-10-09_ppocrv5_det_rec` (detector + recognizer) | 71.1% | 50.2% | 29.8% | 30.2% | 35.2% | 42.5% (45/106) |

  The last row's CER and word accuracy cover only the 106 matched words, so
  they flatter it.
- **Best model:** `2026-10-09_yolo_detector_ppocrv5` — CER 28.4%
  (case-insensitive), R×(1-CER) 57.3%, but not the overall best: its word
  accuracy is below SimpleHTR and TrOCR large. Not the pipeline of record.
- **Conclusion:**
    - As a recogniser, PP-OCRv5 (8 MB, ONNX) has the lowest CER, but the
      gain over SimpleHTR (about 3 points) is likely within noise on 169
      words, and word accuracy is lower. No reason to replace SimpleHTR.
    - As a full pipeline it finds far fewer words than our YOLO detector
      (recall 50.2% vs. 80.1%), so detection is the weak part.
    - Next: the server variants and PP-OCRv6, if worth the effort.

## Current status

- `PPOCRv5Model` implemented with `recognize`, `recognize_batch` and
  `read_page`; both pipelines are wired into `compute_predictions`.
- Benchmarked; see [Experiments](#experiments).
- PP-OCRv5 is trained mostly on printed text, so handwriting accuracy is
  uncertain. The online demo check (stage 1 of issue #157) was done separately.
- `2026-10-09_ppocrv5_det_rec` is not in `CROP_RECORDING_PIPELINES`: it has no
  per-word crops to record.

## Outlook

- Try the server variants and `PP-OCRv6` (both supported by `rapidocr`).
