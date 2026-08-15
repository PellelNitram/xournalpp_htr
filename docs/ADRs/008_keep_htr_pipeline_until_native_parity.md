# ADR 008 – Keep `htr_pipeline` Until the Native Pipeline Reaches Parity

- Date: 2026-08-15
- Status: Accepted
- PRD: None
- Drivers: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))
- Deciders: Martin Lellep ([@PellelNitram](https://github.com/PellelNitram/))

## Context

Removing the external `htr_pipeline` dependency in favour of our own models is
tracked in [#125](https://github.com/PellelNitram/xournalpp_htr/issues/125),
motivated by a simpler installation story.
[#127](https://github.com/PellelNitram/xournalpp_htr/issues/127) benchmarked both
pipelines on `xournalpp_htr_benchmark` (6 documents, 211 ground truth words):

| Metric | `2024-07-18_htr_pipeline` | `2026-06-07_htr_pipeline_native` |
| --- | --- | --- |
| Precision | 77.8% | 84.5% |
| Recall | 69.7% | 41.2% |
| CER (case-insensitive) | 39.4% | 64.9% |

The product goal is **search**, so case-folded metrics are the relevant ones.
Combining both stages as `recall x (1 - CER)` -- the share of ground truth text
surviving detection *and* recognition -- the external pipeline recovers 42.2%
against the native pipeline's 14.5%, roughly three times as much.

## Decision

1. **`2024-07-18_htr_pipeline` remains the default.** #125 stays blocked.
2. **Gate for #125.** The native pipeline may replace it once it matches or beats
   the external pipeline on the benchmark on both **recall (>= 69.7%)** and
   **precision (>= 77.8%)**. It clears precision today (84.5%) and fails recall
   (41.2%). No tolerance is granted -- see Rationale.
3. **Model effort goes to detection, not recognition.** The native pipeline emits
   103 predictions for 211 words, so recall is capped by the word detector.
4. **Supersession.** [#145](https://github.com/PellelNitram/xournalpp_htr/issues/145)
   will choose a primary metric for search-oriented HTR. If it concludes before
   the gate is met, it supersedes this ADR. If the gate is met first, the question
   is moot and #145 governs later evaluation only.

## Rationale

Parity with today's behaviour is the conservative floor: easing installation must
not silently degrade what users can find. An undetected word is permanently
unfindable, so missing 59% of words cannot be hidden behind a flag.

The decision does not depend on the unresolved metric question, because every
candidate metric in #145 is bounded above by recall -- none can rank the native
pipeline ahead while it detects half as many words.

The gate is a **floor, not a claim of statistical parity**. At 211 words the
standard error on recall is around 3 pp, and higher in practice since words
within a document are not independent, so any meaningful tolerance would sit
inside the noise. CER was deliberately not used: `benchmark.py` computes it over
matched pairs only, making it detection-blind and flattering to a pipeline that
under-detects.

## Consequences

### Pros

- Users keep today's recognition quality.
- A pre-committed threshold removes the temptation to rationalise "close enough".
- Detector recall is named as the single blocking metric.
- The PyInstaller question stays independently testable: bundle with the native
  pipeline as the only included path, without removing anything.

### Cons

- The `htr_pipeline` dependency and its installation cost persist.
- Both pipelines must be maintained in `compute_predictions()` meanwhile.
- Weak measurement: 6 documents, one holding 55% of the words, no error bars, and
  likely a single writer -- cross-writer generalisation is untested. Tuning the
  detector against 211 words risks overfitting. Growing the corpus would
  strengthen the gate; not tracked here.
- The threshold may be discarded once #145 lands.

## Alternatives

- **Remove `htr_pipeline` now and accept the regression.** Rejected: 41.2% recall
  leaves most handwriting unsearchable.
- **Wait for #145 first.** Rejected: the ranking does not depend on the metric
  choice, so blocking would have stalled #125 and the model work.
- **Ship native behind an opt-in flag.** Rejected: does not remove the dependency,
  so it buys none of the installation simplification, and adds a third
  configuration to support.
