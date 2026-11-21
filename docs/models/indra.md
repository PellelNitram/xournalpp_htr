# Indra

Datalab's pretrained [Indra OCR](https://github.com/datalab-to/indra)
([weights](https://huggingface.co/datalab-to/indra-ocr)), tried as a page
reader (issue #166). It finds the text lines of a page and reads them, run as
ONNX through the [`indra-ocr`](https://pypi.org/project/indra-ocr/) package: no
PyTorch, no GPU, no server.

**Optional.** It is built for printed documents; handwriting is a stated
weakness. Datalab reports 25.2% on its OmniParseBench handwriting tests and
"about a third of the words on held-out handwritten pages" wrong. On our
benchmark it is nevertheless competitive as a whole-page reader, but it does
not beat the pipeline of record.

**Exception to ADR 006:** the weights are not our own export; `indra-ocr`
downloads them from the HF Hub on first use (about 76 MB for the default set,
of which the Latin path is about 25 MB). The dependency sits behind the
optional `indra` extra.

**License:** the code is Apache-2.0, the weights are under a modified
OpenRAIL-M: free for research, personal use and startups under $20M in funding
or revenue, and not to be used competitively with Datalab's API. Check it
before any commercial use.

One pipeline:

| Pipeline | What it does |
|---|---|
| `2026-11-21_indra_page` | The whole page through Indra's own line detector and recogniser. Lines are cut into words at the recogniser's spaces. No YOLO detector, no stroke grouping. |

## How words are cut

Indra returns lines, not words. Its recogniser (a CTC model) emits a token
sequence per line, including space tokens, each with the x position of its CTC
peak. `IndraModel.read_page` cuts each line at the spaces. A word spans from
its first to its last token, and its box is stretched to the midpoint of the
gap to each neighbouring word (the ends of the line for the first and last
word). It uses the package's engine (`detect`, `recognize`), not its public
`Converter.ocr`, which returns lines only; the extra therefore pins the
version.

## GPU training setup

Not applicable: we have not trained or finetuned Indra.

## Inference

```bash
uv sync --extra indra
```

`uv sync` removes extras that you don't name, so list every extra you use. To
add Indra to a running environment, use `uv run --extra indra ...`.

```python
from xournalpp_htr.inference_models import IndraModel

model = IndraModel.from_pretrained()
words = model.read_page(page_image_grayscale)  # [(text, BoundingBox), ...]
```

Benchmark:

```bash
uv run --extra indra python scripts/run_benchmark.py -p 2026-11-21_indra_page
```

## Best model

`2026-11-21_indra_page` is the only Indra pipeline. Its R×(1-CER) is **57.5%**,
second to `2026-11-03_online_word_detector_hub_simple_htr_ondb` (**63.6%**) among
the pipelines that need no server, and below `2026-11-09_yolo_detector_lfm3_vl`
(58.4%), which needs a `llama-server` and about 20 minutes per benchmark run on
CPU. A run of Indra takes 1 to 2 minutes. See [Experiments](#experiments).

## Experiments

All results use the benchmark dataset `latest` as of 2026-11-21 (406 ground
truth words), on CPU only, as the [LFM3-VL-2B](lfm3_vl.md) experiments do, and
are comparable to them. They are single runs without significance tests. CER and
word accuracy are computed on the matched words only, so rows with different
numbers of matches are not strictly comparable on those two columns.

### 2026-11-21 — Indra as recogniser, as page reader and combined with the online detector

- **Hypothesis:** Indra reads handwriting well enough to be useful in a
  pipeline, either behind a word detector or as a whole-page reader.
- **Setup:** three spikes with throwaway scripts (not kept in the repository),
  `indra-ocr` 1.1.2:
    1. *Crops:* the YOLO detector cuts the words and each crop is read as one
       line box by Indra's recogniser, without its line detector (the
       `2026-10-09_yolo_detector_ppocrv5` pipeline with Indra in place of
       PP-OCRv5). Calling Indra's full pipeline on a crop instead made its
       detector split the word into overlapping fragments (CER about 130%) and
       is not reported.
    2. *Page:* the pipeline in this repository. Two box variants were
       compared: `mid` (boxes stretched to the midpoint of the gaps, the one
       implemented) and `pad` (boxes of the first to the last token plus 0.35
       line heights).
    3. *Combined with the online detector:* the stroke-based detector supplies
       the word boxes and each takes the text of the Indra word that it
       overlaps most. Lines either from Indra's own detector or built from the
       stroke groups.
- **Command:** `uv run --extra indra python scripts/run_benchmark.py -p 2026-11-21_indra_page`
  for the page pipeline (it reproduced its spike exactly).
- **Results:**

| Variant | Precision | Recall | CER (case-insens.) | R×(1-CER) | Word Acc |
|---|---|---|---|---|---|
| Page, `mid` (the pipeline) | 81.1% | 74.9% | 23.2% | **57.5%** | 56.9% (173/304) |
| Page, `pad` | 78.4% | 72.4% | 22.8% | 55.9% | 57.1% |
| Crops behind YOLO | 77.0% | 72.7% | 42.4% | 41.9% | 38.6% |
| Online groups + Indra, Indra's lines | 84.1% | 87.4% | 37.0% | 55.1% | 46.5% |
| Online groups + Indra, lines from groups | 84.1% | 87.4% | 65.0% | 30.6% | 22.5% |

- **Best model:** page, `mid`.
- **Conclusion:**
    - Indra needs the context of a line: on isolated word crops its CER is
      42.4%, nearly double the 23.2% of page mode.
    - As a page reader its detector finds more words than YOLO does (precision
      81.1% and recall 74.9% against 77.0% and 72.7%) and its CER is on a par
      with the best pipelines. The stroke-based online detector still finds
      far more (87.4% recall).
    - The wider `mid` boxes beat the tight `pad` boxes under the benchmark's
      strict 80% coverage match; the 1.6 point gap in R×(1-CER) is within what
      we would expect from noise.
    - Combining the online detector with Indra does not work: Indra's words
      and the stroke groups disagree on where words start and end, and
      assigning by overlap leaves 20 of 422 groups without text and gives many
      wrong ones. The variant with lines built from the stroke groups is
      broken (125 of 422 groups empty) for a reason that was not found: our
      line clustering is crude and the result says nothing about Indra.

### 2026-11-21 — Comparison with the existing pipelines

The pipelines from the [LFM3-VL-2B comparison](lfm3_vl.md#2026-11-21-comparison-with-the-existing-pipelines)
and Indra, sorted by R×(1-CER). The older rows were not rerun.

| Pipeline | Precision | Recall | CER (case-insens.) | R×(1-CER) | Word Acc |
|---|---|---|---|---|---|
| `2026-11-03_online_word_detector_hub_simple_htr_ondb` | 84.1% | 87.4% | 27.3% | **63.6%** | 53.8% |
| `2026-11-09_yolo_detector_lfm3_vl` | 77.0% | 72.7% | **19.6%** | 58.4% | **66.4%** |
| `2026-11-21_indra_page` | 81.1% | 74.9% | 23.2% | 57.5% | 56.9% |
| `2026-10-09_yolo_detector_ppocrv5` | 77.0% | 72.7% | 23.4% | 55.7% | 51.2% |
| `2026-10-17_yolo_detector_simple_htr_ondb_beam_vocab` | 77.0% | 72.7% | 24.6% | 54.8% | 60.3% |
| `2026-10-09_ppocrv5_det_rec` | 77.3% | 51.2% | 20.7% | 40.6% | 57.2% |

Indra is the second best pipeline without a server, and the only whole-page
one that is competitive: PP-OCRv5's page mode finds far fewer words
(recall 51.2%).
