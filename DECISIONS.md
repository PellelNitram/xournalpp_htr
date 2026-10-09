# Decisions — TrOCR experiment (issue #156)

Branch: `156-try-trocr-for-text-transcription`. Decided 2026-10-05.

Goal: find out whether TrOCR transcribes our word crops better than
SimpleHTR. Recognition (CER), not detection, is the bottleneck: word accuracy
stays between 35% and 47% across all pipelines (see the issue #120 notes in
`docs/models/simple_htr.md`).

The decisions below are shortcuts that keep the experiment cheap. **If TrOCR
wins, come back to this file**: several of them break project conventions and
must be revisited before the pipeline counts as the pipeline of record.

## 1. Detector: YOLO

- **Decision:** Pair TrOCR with `YOLOWordDetectorModel`, the detector used by
  `2026-09-02_yolo_detector` and `2026-10-05_yolo_detector_beam_vocab`.
- **Why:** Best detector recall on the benchmark (80.1%, against 66.8% for
  RF-DETR). Holding the detector fixed means any change in CER or word
  accuracy comes from the recognizer alone.

## 2. Separate pipeline: `2026-10-05_yolo_detector_trocr`

- **Decision:** Add a new dated pipeline in `xournalpp_htr/models.py` and leave
  the existing ones alone.
- **Why:** Project rule: experiments that change output get their own pipeline
  so older ones stay reproducible.
- **Baselines to compare against:** `2026-09-02_yolo_detector` (greedy
  SimpleHTR) and `2026-10-05_yolo_detector_beam_vocab` (current best word
  accuracy: CER 31.7%, word accuracy 46.7%).

## 3. Run TrOCR with `transformers`, not ONNX ⚠️ breaks a convention

- **Decision:** `TrOCRModel` (in `xournalpp_htr/inference_models.py`) loads
  the checkpoint with `transformers` (`TrOCRProcessor` and
  `VisionEncoderDecoderModel`) and runs it with `torch`. Those dependencies sit
  behind a new optional `trocr` extra (`uv sync --extra trocr`) and are
  imported lazily, so the base install stays lean.
- **Convention it breaks:** ADR 006 and `docs/models/index.md` require
  inference to run ONNX artifacts from HF Hub with no `transformers`
  dependency.
- **Why:** Measure first. Exporting an encoder-decoder model to ONNX means
  several graphs, a hand-written autoregressive decoding loop and BPE
  tokenizer handling. That is a lot of work before we even know whether
  TrOCR helps.
- **If TrOCR wins:** Export it to ONNX (e.g. with `optimum`: encoder plus
  decoder with past key values), publish it as
  `PellelNitram/xournalpp-htr-trocr`, reimplement generation and detokenization
  without `transformers`, check the outputs match, then drop the `trocr` extra.
  Start that work in a new dated pipeline.

## 4. Checkpoint: `microsoft/trocr-base-handwritten`, used as-is

- **Decision:** Use Microsoft's IAM-finetuned base model (334M parameters)
  with no finetuning of our own, at `revision="main"`.
- **Why:** A sensible balance between quality and speed. `large-handwritten`
  (558M) is the best on IAM but slow on CPU, which matters for the plugin.
  `small-handwritten` (62M) is clearly worse.
- **Caveats:**
  - `HF_REPO_ID` points at Microsoft's repo, so it does not follow the
    `PellelNitram/xournalpp-htr-<model>` naming.
  - `main` can change underneath us. If the result matters, pin a commit
    hash.
  - The checkpoint was finetuned on IAM text *lines*, while we feed it single
    word crops. A mismatch in input distribution is expected.
- **If TrOCR wins:** Try `large-handwritten`, and finetuning on our own word
  crops (rendered online data, #150 and #153).

## 5. Inference details

- **Preprocessing:** Grayscale crop → RGB → the processor's own resize to
  384×384 (aspect ratio not preserved) and normalisation. `preprocess()`
  repeats only the resize, so the crop analysis can show what the encoder sees.
- **Batching:** Each page's crops are recognised together
  (`recognize_batch`, batches of 16). This differs from the SimpleHTR
  pipelines, which recognise one crop at a time.
- **Generation:** The checkpoint's default generation config, capped at
  `MAX_NEW_TOKENS = 32`. Output is stripped of surrounding whitespace and left
  otherwise untouched: TrOCR may produce punctuation and casing that SimpleHTR
  never does. The benchmark CER is case-insensitive, punctuation is not.
- **Device:** CUDA when available, otherwise CPU.
- **No lexicon:** The vocabulary snapping from `beam_vocab` only applies to
  CTC beams. TrOCR has an internal language model and should not need it.

## 6. Where it runs

- **Decision:** Run the benchmark on the GPU VM `martin-l4` (L4, in
  `~/xournalpp_htr`), syncing code via git (commit, push, pull). The MacBook
  has no network, so it cannot download the models or update `uv.lock`;
  re-lock on the VM. Code only flows one way: commit on the MacBook, push to
  GitHub, `git pull` on the VM, run. Never commit on the VM. A file the VM
  generates (`uv.lock`) is copied back, committed locally and pushed. (The
  first two lock commits were made on the VM and fetched over ssh, before this
  rule.) Sync with `uv sync --all-extras` so the other extras installed on the
  VM aren't removed.
- **Command:** `uv run python scripts/run_benchmark.py -p
  2026-10-05_yolo_detector_trocr`.

## Implementation notes from the run

- transformers 5 can't auto-load this checkpoint's tokenizer (the repo has
  only `vocab.json` and `merges.txt`, no `tokenizer.json`; the error message
  wrongly points at `sentencepiece`). `TrOCRModel.from_pretrained` therefore
  builds `TrOCRProcessor` from `AutoImageProcessor` and `RobertaTokenizer`
  directly.
- The load report flags `encoder.pooler.dense.*` as newly initialised. This is
  harmless: TrOCR's decoder never uses the ViT pooler.
- Re-locking on the VM also caught up `uv.lock` entries that earlier extras
  (`pyctcdecode`, `rfdetr`, `ultralytics`, `english-words`) had never locked.

## Results (2026-10-05, `martin-l4`, commit `37749d0`)

`uv run python scripts/run_benchmark.py -p 2026-10-05_yolo_detector_trocr
--crop-analysis crop_analysis_trocr` (benchmark dataset `latest`; crop
analysis in `martin-l4:~/xournalpp_htr/crop_analysis_trocr/`).

| Pipeline | CER (case-insens.) | CER (case-sens.) | R×(1-CER) | Word Acc |
|---|---|---|---|---|
| `2026-09-02_yolo_detector` (SimpleHTR, greedy) | 34.4% | — | 52.5% | 39.1% (66/169) |
| `2026-10-05_yolo_detector_beam_vocab` (SimpleHTR, beam + vocab) | **31.7%** | — | **54.7%** | **46.7% (79/169)** |
| `2026-10-05_yolo_detector_trocr` (TrOCR base, as-is) | 50.8% | 61.0% | 39.4% | 29.6% (50/169) |

Precision and recall are unchanged (73.8%, 80.1%), as expected with the same
detector.

Error analysis (from `manifest.csv`, 119 wrong matched words):

- **Line-model artefacts:** 57 predictions contain a space, mostly a
  sentence-final ` .` or ` ,` appended to a single word (`'stroke' → 'quote
  .'`). This is IAM *line* finetuning showing through on word crops.
- **Hallucinated or garbled words:** most remaining errors are fluent but wrong
  or invented words (`'chiefly' → 'alvely'`, `'is' → '13.'`, `'it' → 'yfy'`),
  often longer than the GT (on average 1.8 characters longer).
- **Only 18 of the 119 errors** go away when punctuation and spaces are
  ignored.

Offline estimate with post-processing (same matched words, case-insensitive):

| TrOCR output | CER | Word Acc |
|---|---|---|
| Raw (reproduces the benchmark) | 50.8% | 29.6% |
| Spaced trailing punctuation stripped | 43.7% | 34.3% |
| All spaces and trailing punctuation dropped (generous) | 38.9% | 37.9% |

**Conclusion:** off-the-shelf `trocr-base-handwritten` on word crops loses to
SimpleHTR, even with generous cleanup: 38.9% CER against 34.4% for greedy
SimpleHTR and 31.7% for beam + vocab. The experiment did **not** work well,
so the convention-breaking decisions above (3, 4) stay confined to this
pipeline. Do not port it to ONNX yet.

Possible follow-ups, if TrOCR is revisited:

1. **Line-level input:** feed TrOCR whole text lines (word boxes clustered into
   lines) instead of word crops, matching what it was finetuned on. Splitting
   the line transcription back onto word boxes is the hard part.
2. **Finetune** on our own word crops (rendered online data, #150 and #153),
   so the model stops expecting sentence context.
3. **`trocr-large-handwritten`**: cheap to try (change `HF_REPO_ID`), but
   unlikely to fix the line-versus-word mismatch. *Tried below; it helped a
   lot more than expected.*

## 7. Large checkpoint: `2026-10-05_yolo_detector_trocr_large`

- **Decision:** Add a second pipeline with
  `microsoft/trocr-large-handwritten` (558M parameters), the largest
  handwritten TrOCR checkpoint. Everything else (detector, preprocessing,
  generation, no post-processing) is identical to the base pipeline, so the
  two isolate the effect of model size.
- **Implementation:** `TrOCRLargeModel(TrOCRModel)` only overrides
  `HF_REPO_ID`; the large repo uses the same RoBERTa tokenizer layout, so the
  explicit tokenizer workaround applies unchanged. Both pipeline names share
  one branch in `compute_predictions`.

## Results: large checkpoint (2026-10-05, `martin-l4`, commit `3e8bb63`)

`uv run python scripts/run_benchmark.py -p 2026-10-05_yolo_detector_trocr_large
--crop-analysis crop_analysis_trocr_large`.

| Pipeline | CER (case-insens.) | CER (case-sens.) | R×(1-CER) | Word Acc |
|---|---|---|---|---|
| `2026-09-02_yolo_detector` (SimpleHTR, greedy) | 34.4% | — | 52.5% | 39.1% (66/169) |
| `2026-10-05_yolo_detector_beam_vocab` (SimpleHTR, beam + vocab) | **31.7%** | — | **54.7%** | **46.7% (79/169)** |
| `2026-10-05_yolo_detector_trocr` (TrOCR base) | 50.8% | 61.0% | 39.4% | 29.6% (50/169) |
| `2026-10-05_yolo_detector_trocr_large` (TrOCR large) | 42.1% | 50.7% | 46.4% | 37.3% (63/169) |

Large is far better than base, but on the raw benchmark it still trails both
SimpleHTR pipelines. Its errors are dominated by the line-model artefact:
66 of 106 wrong words contain a space, typically an invented trailing
` .` or ` ,` on an otherwise correct word (`'words' → 'words .'`,
`'issues' → 'issues .'`, `'itself' → 'itself .'`).

Offline estimate with post-processing (same matched words, case-insensitive):

| Post-processing | TrOCR base CER / Word Acc | TrOCR large CER / Word Acc |
|---|---|---|
| None (reproduces the benchmark) | 50.8% / 29.6% | 42.1% / 37.3% |
| IAM detokenization (drop space before punctuation) | 46.8% / 29.6% | 34.9% / 38.5% |
| **Strip spaced trailing punctuation** | 43.7% / 34.3% | **29.3% / 48.5%** |
| Also drop all spaces and trailing punctuation (generous) | 38.9% / 37.9% | 27.3% / 51.5% |

The regex for the stripping variant is `(\s+[.,;:!?'"]+)+$`, removed from the
end of the prediction. Detokenization barely helps because the punctuation is
invented, not just badly spaced.

**Conclusion (offline):** with a simple clean-up (strip punctuation that
TrOCR appends after a space), TrOCR large is estimated at **29.3% CER and
48.5% word accuracy**, on par with the current best,
`2026-10-05_yolo_detector_beam_vocab` (31.7%, 46.7%). Confirmed below in a
real pipeline.

## 8. Stripping pipeline: `2026-10-09_yolo_detector_trocr_large_strip_punct`

- **Decision:** Add a pipeline that runs TrOCR large and then removes
  punctuation appended after a space (`strip_appended_punctuation` in
  `xournalpp_htr/inference_models.py`, regex `(\s+[.,;:!?'"]+)+$`).
  Punctuation attached to a word (`thoughts:`) is kept. The function sits
  outside `TrOCRModel`, so it can be unit-tested without `transformers`.
- **Only large:** stripping helps base too (estimated 50.8% → 43.7% CER), but
  base stays behind plain greedy SimpleHTR either way, so there is no base
  stripping pipeline.
- **Why ship it at all:** not because it is clearly better (see the
  assessment below), but so users who want TrOCR can choose it.

## Results: stripping pipeline (2026-10-09, `martin-l4`, commit `8b4791b`)

`uv run python scripts/run_benchmark.py -p
2026-10-09_yolo_detector_trocr_large_strip_punct --crop-analysis
crop_analysis_trocr_large_strip_punct`. It reproduces the offline estimate
exactly.

| Pipeline | CER (case-insens.) | CER (case-sens.) | R×(1-CER) | Word Acc |
|---|---|---|---|---|
| `2026-09-02_yolo_detector` (SimpleHTR, greedy) | 34.4% | — | 52.5% | 39.1% (66/169) |
| `2026-10-05_yolo_detector_beam_vocab` (SimpleHTR, beam + vocab) | 31.7% | — | 54.7% | 46.7% (79/169) |
| `2026-10-05_yolo_detector_trocr` (TrOCR base) | 50.8% | 61.0% | 39.4% | 29.6% (50/169) |
| `2026-10-05_yolo_detector_trocr_large` (TrOCR large) | 42.1% | 50.7% | 46.4% | 37.3% (63/169) |
| `2026-10-09_yolo_detector_trocr_large_strip_punct` (TrOCR large + strip) | **29.3%** | 37.9% | **56.6%** | **48.5% (82/169)** |

## Assessment: not convinced (2026-10-09)

The stripping pipeline has the best numbers on paper, but that is not enough
to make it the pipeline of record:

- **The gain is within noise.** 82 vs. 79 correct words out of 169. The
  standard error of word accuracy on 169 words is about ±3.8 points, so the
  1.8-point gap is not meaningful. The CER gap (29.3% vs. 31.7%) is small too.
  The benchmark dataset is small, and the gain may not carry over.
- **The cost is large.** 558M parameters vs. 3.2M for SimpleHTR (~170×);
  2.2 GB of float32 weights vs. a 13 MB ONNX file. TrOCR also decodes
  autoregressively (one decoder pass per output token), so it is much slower,
  especially on plugin users' CPUs.

| Model | Parameters (encoder / decoder) | Weights |
|---|---|---|
| SimpleHTR | 3.2M | 13 MB (ONNX) |
| TrOCR base | 334M (87M / 247M) | 1.3 GB (float32) |
| TrOCR large | 558M (305M / 254M) | 2.2 GB (float32) |

**Decision:** keep `2026-10-05_yolo_detector_beam_vocab` as the pipeline of
record and offer `2026-10-09_yolo_detector_trocr_large_strip_punct` as an
optional pipeline (needs the `trocr` extra). The convention-breaking decisions
3 and 4 stay as they are; an ONNX port is not justified by this result.

What would change the decision:

1. **A larger benchmark** that shows a clear, significant gap in TrOCR's
   favour.
2. **Finetuning on our own word crops** (#150, #153), which should remove the
   punctuation habit without post-processing and attack the remaining
   garbled words (`'chiefly' → 'clmolly .'`).
3. **Line-level input**, matching what the checkpoints were finetuned on.
4. If any of these makes TrOCR the clear winner: measure CPU latency per page
   and try float16 or int8 weights before an ONNX port.
