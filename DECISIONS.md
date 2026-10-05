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
  re-lock on the VM. The VM can't push to GitHub, so commits made there are
  fetched over ssh (`git pull martin-l4:xournalpp_htr <branch>`) and pushed
  from the MacBook. Sync with `uv sync --all-extras` so the other extras
  installed on the VM aren't removed.
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
   unlikely to fix the line-versus-word mismatch.
