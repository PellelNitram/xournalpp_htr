# TrOCR

Word-level text recognition with Microsoft's pretrained
[TrOCR](https://huggingface.co/docs/transformers/en/model_doc/trocr), tried as
an alternative to [SimpleHTR](simple_htr.md) (issue #156). We use the
IAM-finetuned checkpoints
[`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten)
and
[`microsoft/trocr-large-handwritten`](https://huggingface.co/microsoft/trocr-large-handwritten)
as they are, with the YOLO word detector
([WordDetectorYOLO](word_detector_yolo.md)) in front.

**Exception to ADR 006:** unlike our other models, TrOCR runs through
`transformers` and `torch` instead of an ONNX export from our own HF Hub repo.
The dependencies sit behind the optional `trocr` extra. See
[Current status](#current-status) for why, and [Outlook](#outlook) for when to
revisit it.

To use the TrOCR pipeline for your own notes, see the
[User Guide](../user_guide.md#optional-trocr-pipeline).

## GPU training setup

Not applicable: we have not trained or finetuned TrOCR. Finetuning on our own
word crops is listed under [Outlook](#outlook).

## Inference

```bash
uv sync --extra trocr
```

```python
from xournalpp_htr.inference_models import (
    TrOCRLargeModel,
    strip_appended_punctuation,
)

model = TrOCRLargeModel.from_pretrained()  # or TrOCRModel for base
texts = model.recognize_batch(word_images_grayscale)
texts = [strip_appended_punctuation(text) for text in texts]
```

- `from_pretrained()` loads a pinned commit (`REVISION`) rather than `main`,
  because we don't control Microsoft's repos.
- Inference uses CUDA when available, otherwise CPU.
- `strip_appended_punctuation` removes punctuation that TrOCR appends after a
  space (`"words ."` → `"words"`); punctuation attached to a word
  (`"thoughts:"`) is kept.

## Best model

TrOCR large with punctuation stripping, pipeline
`2026-10-09_yolo_detector_trocr_large_strip_punct`: **CER 29.3%
(case-insensitive), word accuracy 48.5%**.

This is **not** the project's pipeline of record.
`2026-10-05_yolo_detector_beam_vocab` (SimpleHTR, CER 31.7%, word accuracy
46.7%) remains the default:

- **The gain is within noise.** 82 vs. 79 correct words out of 169; the
  standard error of word accuracy on 169 words is about ±3.8 points.
- **The cost is large:**

| Model | Parameters (encoder / decoder) | Weights | CPU time per word* |
|---|---|---|---|
| SimpleHTR | 3.2M | 13 MB (ONNX) | not measured |
| TrOCR base | 334M (87M / 247M) | 1.3 GB (float32) | ~1.4 s |
| TrOCR large | 558M (305M / 254M) | 2.2 GB (float32) | ~2.9 s |

\* 4 vCPUs, Intel Xeon @ 2.2 GHz; 7 benchmark pages, ~229 word crops. The
slowest page took 5.5 minutes with TrOCR large.

## Experiments

All runs use the benchmark dataset (`latest`) and the same YOLO detector, so
precision (73.8%) and recall (80.1%) are identical; only recognition changes.
Command: `uv run python scripts/run_benchmark.py -p <pipeline>`.

| Pipeline | CER (case-insens.) | CER (case-sens.) | R×(1-CER) | Word Acc |
|---|---|---|---|---|
| `2026-09-02_yolo_detector` (SimpleHTR, greedy) | 34.4% | — | 52.5% | 39.1% (66/169) |
| `2026-10-05_yolo_detector_beam_vocab` (SimpleHTR, beam + vocab) | 31.7% | — | 54.7% | 46.7% (79/169) |
| `2026-10-05_yolo_detector_trocr` (TrOCR base) | 50.8% | 61.0% | 39.4% | 29.6% (50/169) |
| `2026-10-05_yolo_detector_trocr_large` (TrOCR large) | 42.1% | 50.7% | 46.4% | 37.3% (63/169) |
| `2026-10-09_yolo_detector_trocr_large_strip_punct` (large + strip) | **29.3%** | 37.9% | **56.6%** | **48.5% (82/169)** |

### 2026-10-09 — TrOCR large with punctuation stripping

- **Hypothesis:** stripping the punctuation TrOCR appends after a space
  recovers most of large's errors (an offline re-scoring estimated 29.3% CER).
- **Setup:** pipeline `2026-10-09_yolo_detector_trocr_large_strip_punct`,
  commit `8b4791b`. Regex `(\s+[.,;:!?'"]+)+$` removed from the end of each
  prediction. Only large: stripping helps base too (50.8% → 43.7% CER,
  offline), but base stays behind greedy SimpleHTR.
- **Results:** CER 29.3%, word accuracy 48.5%, matching the offline estimate
  exactly. After pinning the checkpoints (commit `c5311d5`), all three TrOCR
  pipelines reproduce their numbers exactly, on GPU and CPU.
- **Best model:** this pipeline; see [Best model](#best-model) for why it is
  not the project default.
- **Conclusion:** on par with SimpleHTR beam + vocab at ~170× the size. Ship
  as an optional pipeline only.

### 2026-10-05 — TrOCR large

- **Hypothesis:** the larger checkpoint reads our handwriting better than base.
- **Setup:** pipeline `2026-10-05_yolo_detector_trocr_large`, commit
  `3e8bb63`; otherwise identical to the base run.
- **Results:** CER 42.1%, word accuracy 37.3%. 66 of 106 wrong words contain
  a space, mostly an invented trailing ` .` or ` ,` on an otherwise correct
  word (`'words' → 'words .'`). Moving the space instead (`words.`) barely
  helps (34.9% CER), because the punctuation is invented.
- **Best model:** TrOCR large, raw output.
- **Conclusion:** much better than base. The remaining gap to SimpleHTR is
  mostly the appended punctuation, which led to the stripping run.

### 2026-10-05 — TrOCR base, as-is

- **Hypothesis:** an off-the-shelf transformer recognizer beats SimpleHTR on
  our word crops, since recognition (CER), not detection, limits word accuracy.
- **Setup:** pipeline `2026-10-05_yolo_detector_trocr`, commit `37749d0`.
  Crops are resized to 384×384 by the processor and recognised in batches of
  16; default generation, at most 32 new tokens; no post-processing.
- **Results:** CER 50.8%, word accuracy 29.6%. 57 of 119 wrong words contain
  a space (appended ` .`/` ,`), and most others are fluent but invented words
  (`'chiefly' → 'alvely'`, `'is' → '13.'`).
- **Best model:** TrOCR base, raw output.
- **Conclusion:** clearly worse than SimpleHTR. The checkpoints were finetuned
  on IAM text *lines*, and that shows on single word crops.

## Current status

- [x] `TrOCRModel` and `TrOCRLargeModel` in `xournalpp_htr/inference_models.py`;
  three pipelines in `xournalpp_htr/models.py`; `trocr` extra.
- **Why `transformers` instead of ONNX:** measuring first was cheaper than
  exporting an encoder-decoder model, writing a generation loop and handling
  the BPE tokenizer before knowing whether TrOCR helps. Whose model it is
  doesn't matter: ADR 006 is about what runs on the user's machine.
- **Pinning:** `TrOCRModel.REVISION` (base `eaacaf45…`) and
  `TrOCRLargeModel.REVISION` (large `e68501f4…`) are the commits of `main` on
  2026-10-05. The `trocr` extra requires `transformers>=5.17,<6`.
- **Known issues:**
  - transformers 5 can't auto-load the tokenizer of these older repos
    (`vocab.json` + `merges.txt`, no `tokenizer.json`; the error wrongly
    blames `sentencepiece`), so `from_pretrained()` builds the processor from
    `AutoImageProcessor` and `RobertaTokenizer` directly.
  - The load report flags `encoder.pooler.dense.*` as newly initialised. This
    is harmless: the decoder never uses the encoder's pooler.
  - For the large model, `transformers` takes `model.safetensors` from Hugging
    Face's automatic conversion pull request (`refs/pr/9`), not from `main`.
    The weights are identical either way.
  - The Lua plugin always uses the default pipeline, so TrOCR is available
    from the command line only.

## Outlook

Port TrOCR to ONNX only if it becomes something plugin users should get
easily, for example with the installation work in December 2026 or after one
of these makes it a clear winner:

1. **A larger benchmark** showing a clear, significant gap in TrOCR's favour.
2. **Finetuning on our own word crops** (#150, #153), which should remove the
   punctuation habit and the invented words (`'chiefly' → 'clmolly .'`).
3. **Line-level input**, matching what the checkpoints were finetuned on;
   splitting line transcriptions back onto word boxes is the hard part.

A port would export the encoder and decoder (e.g. with `optimum`), implement
greedy generation with past key values and BPE detokenization in numpy, and
publish the result as `PellelNitram/xournalpp-htr-trocr` (check the license
first). It would remove the 1–2 GB `torch`/`transformers` dependency, and int8
weights would shrink large from 2.2 GB to roughly 600 MB. Measure CPU latency
before and after. The port gets its own dated pipeline.
