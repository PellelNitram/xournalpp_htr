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

Not yet benchmarked.

## Experiments

### 2026-10-09 — PP-OCRv5 as recogniser and as full pipeline

- **Hypothesis:** PP-OCRv5 beats SimpleHTR (CER 31.7%, word accuracy 46.7%,
  `2026-10-05_yolo_detector_beam_vocab`) on handwritten word crops, and
  possibly works without our YOLO detector.
- **Setup:** benchmark dataset `latest`; code revision: fill in.
- **Command:** see [Inference](#inference).
- **Results:** TODO.
- **Best model:** TODO.
- **Conclusion:** TODO.

## Current status

- `PPOCRv5Model` implemented with `recognize`, `recognize_batch` and
  `read_page`; both pipelines are wired into `compute_predictions`.
- Not yet benchmarked.
- PP-OCRv5 is trained mostly on printed text, so handwriting accuracy is
  uncertain. The online demo check (stage 1 of issue #157) was done separately.
- `2026-10-09_ppocrv5_det_rec` is not in `CROP_RECORDING_PIPELINES`: it has no
  per-word crops to record.

## Outlook

- Fill in the benchmark results above.
- Try the server variants and `PP-OCRv6` (both supported by `rapidocr`).
