import base64
import html
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from xournalpp_htr.documents import get_document
from xournalpp_htr.models import PageIndex, WordPrediction, compute_predictions
from xournalpp_htr.xio import load_benchmark

# Annotation classes that carry a text transcription (per ground_truth.schema.json).
_TEXT_CLASSES = {"word", "digit", "mathematical_expression"}

# A prediction matches a ground truth word when it covers most of that word and
# is not wildly larger than it.
#
# IoU is deliberately not used as the criterion. Word detectors pad their boxes
# (see the `margin` in `compute_predictions`), and because IoU divides by the
# union, that padding costs little on long words but is fatal on short ones: a
# perfectly centred box around "is" scores ~0.2 purely because the word is
# small. Coverage asks the question we actually care about -- did the pipeline
# find this word? -- and is unaffected by padding. The area cap stops a single
# oversized box, such as one spanning a whole line, from counting as a match.
_MIN_COVERAGE = 0.8  # fraction of the GT box that must lie inside the prediction
_MAX_AREA_RATIO = 8.0  # prediction area may exceed the GT area by at most this

# Resolution at which page images are rendered into the HTML report.
_REPORT_DPI = 150

Box = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


@dataclass
class GroundTruthWord:
    text: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    page_index: int


@dataclass
class BenchmarkResult:
    precision: float
    recall: float
    cer: float
    cer_case_insensitive: float
    recall_times_accuracy: float
    word_accuracy: float
    n_gt_words: int
    n_predicted_words: int
    n_matched: int
    #: Only populated when `run_benchmark` is called with `collect_details=True`.
    details: "ReportData | None" = None


@dataclass
class WordRow:
    """A ground truth word, a prediction, or a matched pair thereof."""

    #: "correct"  -- matched and the text is exactly right
    #: "wrong"    -- matched but the text differs from the ground truth
    #: "missed"   -- ground truth word without a matching prediction
    #: "spurious" -- prediction without a matching ground truth word
    status: str
    gt_text: str | None
    pred_text: str | None
    gt_box: Box | None
    pred_box: Box | None


@dataclass
class PageDetail:
    page_index: int
    width: float  # in document units (72 DPI)
    height: float
    image_uri: str  # base64 data URI of the rendered page
    rows: list[WordRow]


@dataclass
class SampleDetail:
    name: str
    pages: list[PageDetail]


@dataclass
class ReportData:
    """Everything the HTML report needs beyond the aggregate metrics."""

    pipeline_name: str
    git_sha: str
    created_at: str
    samples: list[SampleDetail] = field(default_factory=list)


def _load_gt_words(gt_path, document) -> list[GroundTruthWord]:
    with open(gt_path) as f:
        gt = json.load(f)

    words = []
    for ann in gt["annotations"]:
        if ann["class"] not in _TEXT_CLASSES:
            continue
        page_index = ann["page_index"]
        layer_index = ann["layer_index"]
        layer = document.pages[page_index].layers[layer_index]
        xs, ys = [], []
        for idx in ann["stroke_indices"]:
            stroke = layer.strokes[idx]
            xs.extend(stroke.x.tolist())
            ys.extend(stroke.y.tolist())
        words.append(
            GroundTruthWord(
                text=ann["text"],
                xmin=float(np.min(xs)),
                xmax=float(np.max(xs)),
                ymin=float(np.min(ys)),
                ymax=float(np.max(ys)),
                page_index=page_index,
            )
        )
    return words


def _iou(a: GroundTruthWord, b: WordPrediction) -> float:
    """Intersection over union. Ranks candidate matches, it does not accept them."""
    ix1 = max(a.xmin, b.xmin)
    iy1 = max(a.ymin, b.ymin)
    ix2 = min(a.xmax, b.xmax)
    iy2 = min(a.ymax, b.ymax)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0.0:
        return 0.0
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    return intersection / (area_a + area_b - intersection)


def _matches(a: GroundTruthWord, b: WordPrediction) -> bool:
    """Whether a prediction covers enough of a GT word to count as finding it."""
    ix1 = max(a.xmin, b.xmin)
    iy1 = max(a.ymin, b.ymin)
    ix2 = min(a.xmax, b.xmax)
    iy2 = min(a.ymax, b.ymax)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    if area_a == 0.0:
        return False
    return intersection / area_a >= _MIN_COVERAGE and area_b / area_a <= _MAX_AREA_RATIO


def _cer(reference: str, hypothesis: str) -> float:
    """Character error rate between two strings via edit distance."""
    r, h = list(reference), list(hypothesis)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / max(len(r), 1)


def _match(
    gt_words: list[GroundTruthWord],
    predictions: dict[PageIndex, list[WordPrediction]],
) -> list[tuple[GroundTruthWord, WordPrediction]]:
    """Greedily pair ground truth words with predictions, one to one per page.

    Acceptable pairs are decided by `_matches`; among those, the pair with the
    tightest fit (highest IoU) is taken first so that a prediction goes to the
    word it sits on rather than to a neighbour it merely overlaps.
    """
    candidates = []
    for gt in gt_words:
        for pred in predictions.get(gt.page_index, []):
            if _matches(gt, pred):
                candidates.append((_iou(gt, pred), gt, pred))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt, matched_pred = set(), set()
    pairs = []
    for _, gt, pred in candidates:
        if id(gt) not in matched_gt and id(pred) not in matched_pred:
            pairs.append((gt, pred))
            matched_gt.add(id(gt))
            matched_pred.add(id(pred))
    return pairs


def run_benchmark(
    pipeline_name: str,
    collect_details: bool = False,
    dataset_version: str | None = None,
) -> BenchmarkResult:
    """Benchmark `pipeline_name` against the xournalpp_htr_benchmark dataset.

    :param pipeline_name: Pipeline to run, see `compute_predictions`.
    :param collect_details: Additionally collect the per-page data the HTML
        report is built from, including a render of every page. This is
        considerably slower, hence opt-in. Stored in `BenchmarkResult.details`.
    :param dataset_version: Git tag or commit hash selecting the dataset
        revision. ``None`` (the default) uses the latest version.
    """
    samples = load_benchmark(dataset_version=dataset_version)

    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_edit_chars = 0
    total_edit_chars_case_insensitive = 0
    total_gt_chars_matched = 0
    total_exact_matches = 0

    details = (
        ReportData(
            pipeline_name=pipeline_name,
            git_sha=_git_sha(),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        if collect_details
        else None
    )

    for sample in samples:
        document = get_document(sample.xopp_path)
        gt_words = _load_gt_words(sample.gt_path, document)
        predictions = compute_predictions(pipeline_name, document)

        n_pred = sum(len(v) for v in predictions.values())
        pairs = _match(gt_words, predictions)

        total_gt += len(gt_words)
        total_pred += n_pred
        total_matched += len(pairs)

        for gt_word, pred_word in pairs:
            total_gt_chars_matched += len(gt_word.text)
            total_edit_chars += round(
                _cer(gt_word.text, pred_word.text) * len(gt_word.text)
            )
            cer_ci = _cer(gt_word.text.lower(), pred_word.text.lower())
            total_edit_chars_case_insensitive += round(cer_ci * len(gt_word.text))
            if cer_ci == 0.0:
                total_exact_matches += 1

        if details is not None:
            details.samples.append(
                SampleDetail(
                    name=sample.xopp_path.name,
                    pages=_collect_pages(document, gt_words, predictions, pairs),
                )
            )

    precision = total_matched / total_pred if total_pred > 0 else 0.0
    recall = total_matched / total_gt if total_gt > 0 else 0.0
    cer = (
        total_edit_chars / total_gt_chars_matched if total_gt_chars_matched > 0 else 0.0
    )
    cer_case_insensitive = (
        total_edit_chars_case_insensitive / total_gt_chars_matched
        if total_gt_chars_matched > 0
        else 0.0
    )

    recall_times_accuracy = recall * (1 - cer_case_insensitive)
    word_accuracy = total_exact_matches / total_matched if total_matched > 0 else 0.0

    return BenchmarkResult(
        precision=precision,
        recall=recall,
        cer=cer,
        cer_case_insensitive=cer_case_insensitive,
        recall_times_accuracy=recall_times_accuracy,
        word_accuracy=word_accuracy,
        n_gt_words=total_gt,
        n_predicted_words=total_pred,
        n_matched=total_matched,
        details=details,
    )


# --- Detail collection -------------------------------------------------------


def _git_sha() -> str:
    """Short SHA of the checkout this code lives in, or "unknown"."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _render_page_uri(document, page_index: int) -> str:
    """Render a page to a PNG and return it as a base64 data URI."""
    with tempfile.NamedTemporaryFile(
        delete=False, prefix="xournalpp_htr__report__", suffix=".png"
    ) as tmpfile:
        path = Path(tmpfile.name)
    try:
        document.save_page_as_image(page_index, path, False, dpi=_REPORT_DPI)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        path.unlink(missing_ok=True)
    return f"data:image/png;base64,{encoded}"


def _collect_pages(
    document,
    gt_words: list[GroundTruthWord],
    predictions: dict[PageIndex, list[WordPrediction]],
    pairs: list[tuple[GroundTruthWord, WordPrediction]],
) -> list[PageDetail]:
    """Build the per-page detail records, rendering each page image."""
    matched_gt = {id(gt): pred for gt, pred in pairs}
    matched_pred = {id(pred) for _, pred in pairs}

    pages = []
    for page_index in range(len(document.pages)):
        rows: list[WordRow] = []

        for gt_word in (w for w in gt_words if w.page_index == page_index):
            pred_word = matched_gt.get(id(gt_word))
            if pred_word is None:
                status = "missed"
            else:
                status = "correct" if pred_word.text == gt_word.text else "wrong"
            rows.append(
                WordRow(
                    status=status,
                    gt_text=gt_word.text,
                    pred_text=None if pred_word is None else pred_word.text,
                    gt_box=(gt_word.xmin, gt_word.ymin, gt_word.xmax, gt_word.ymax),
                    pred_box=None
                    if pred_word is None
                    else (
                        pred_word.xmin,
                        pred_word.ymin,
                        pred_word.xmax,
                        pred_word.ymax,
                    ),
                )
            )

        for pred_word in predictions.get(page_index, []):
            if id(pred_word) in matched_pred:
                continue
            rows.append(
                WordRow(
                    status="spurious",
                    gt_text=None,
                    pred_text=pred_word.text,
                    gt_box=None,
                    pred_box=(
                        pred_word.xmin,
                        pred_word.ymin,
                        pred_word.xmax,
                        pred_word.ymax,
                    ),
                )
            )

        meta = document.pages[page_index].meta_data
        pages.append(
            PageDetail(
                page_index=page_index,
                width=float(meta["width"]),
                height=float(meta["height"]),
                image_uri=_render_page_uri(document, page_index),
                rows=rows,
            )
        )
    return pages


# --- HTML report -------------------------------------------------------------


_REPORT_CSS = """
:root {
  --bg: #ffffff; --fg: #1b1b1f; --muted: #6b6b76; --line: #e2e2e8;
  --correct: #1a7f37; --wrong: #bc4c00; --missed: #d1242f; --spurious: #8250df;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14141a; --fg: #eaeaf0; --muted: #9a9aa6; --line: #2c2c36;
    --correct: #3fb950; --wrong: #ffa657; --missed: #ff7b72; --spurious: #d2a8ff;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1.5rem 5rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .3rem; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .3rem; padding-top: .8rem;
  border-top: 1px solid var(--line); }
h3 { font-size: .9rem; font-weight: 600; color: var(--muted); margin: 1.2rem 0 .4rem; }
.meta { color: var(--muted); font-size: .85rem; }
.legend { display: flex; flex-wrap: wrap; gap: 1.2rem; font-size: .8rem;
  color: var(--muted); margin: 1rem 0; }
.legend > span { position: relative; }
.swatch { display: inline-block; width: .8rem; height: .8rem; border-radius: 2px;
  margin-right: .3rem; vertical-align: -1px; }
.info { display: inline-flex; align-items: center; justify-content: center;
  width: 1rem; height: 1rem; margin-left: .35rem; border-radius: 50%;
  border: 1px solid var(--line); color: var(--muted); font-size: .7rem;
  font-style: italic; font-weight: 700; cursor: help; vertical-align: -1px; }
.info:hover, .info:focus { border-color: var(--fg); color: var(--fg); outline: none; }
.tip { position: absolute; left: 0; top: calc(100% + .4rem); z-index: 10;
  width: min(24rem, 80vw); padding: .6rem .7rem; border: 1px solid var(--line);
  border-radius: 6px; background: var(--bg); color: var(--fg); font-size: .8rem;
  line-height: 1.45; box-shadow: 0 6px 18px rgba(0, 0, 0, .18);
  visibility: hidden; opacity: 0; transition: opacity .12s ease; }
.legend > span:hover .tip, .info:focus + .tip { visibility: visible; opacity: 1; }
.page { position: relative; border: 1px solid var(--line); border-radius: 6px;
  overflow: hidden; background: #fff; }
.page img { display: block; width: 100%; height: auto; }
.page svg { position: absolute; inset: 0; width: 100%; height: 100%; }
svg rect { fill: none; stroke-width: .8; }
svg text { font-size: 7px; font-family: monospace; }
svg .correct rect { stroke: var(--correct); }
svg .correct text { fill: var(--correct); }
svg .wrong rect { stroke: var(--wrong); }
svg .wrong text { fill: var(--wrong); }
svg .spurious rect { stroke: var(--spurious); stroke-dasharray: 3 2; }
svg .spurious text { fill: var(--spurious); }
svg .missed line { stroke: var(--missed); stroke-width: 1.5; }
"""


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _legend() -> str:
    entries = [
        (
            "var(--correct)",
            "prediction, text correct",
            f"The pipeline found this word -- its box contains at least "
            f"{_MIN_COVERAGE:.0%} of the annotated word -- and read it exactly "
            f"right. The comparison is case-sensitive, so &quot;Hello&quot; "
            f"predicted as &quot;hello&quot; is counted as wrong, not correct.",
        ),
        (
            "var(--wrong)",
            "prediction, text wrong",
            "The pipeline found this word but misread at least one character. "
            "Detection is fine here, recognition is not: these words are the ones "
            "that drive the character error rate (CER). Hover the box on the page "
            "to compare the ground truth against the prediction.",
        ),
        (
            "var(--spurious)",
            "prediction without a ground truth word",
            f"The pipeline predicted a word here, but no annotated word matches it. "
            f"Either it fired on something that is not text -- only annotations of "
            f"class word, digit or mathematical expression are ground truth, so "
            f"drawings and separators are not -- or its box misses part of the "
            f"word (less than {_MIN_COVERAGE:.0%} covered) or is more than "
            f"{_MAX_AREA_RATIO:.0f}x larger than it. These lower precision.",
        ),
        (
            "var(--missed)",
            "ground truth word without a prediction (underlined)",
            f"An annotated word the pipeline did not return, drawn as an underline "
            f"beneath the handwriting because there is no prediction box to show. "
            f"These lower recall. An underline sitting inside a dashed box means "
            f"the word was detected but the box does not cover {_MIN_COVERAGE:.0%} "
            f"of it, so it counts as both a miss and a spurious prediction.",
        ),
    ]
    items = "".join(
        f'<span><span class="swatch" style="background:{color}"></span>{label}'
        f'<span class="info" tabindex="0" role="button" aria-label="More information">'
        f'i</span><span class="tip" role="tooltip">{tip}</span></span>'
        for color, label, tip in entries
    )
    return f'<div class="legend">{items}</div>'


def _page_figure(page: PageDetail) -> str:
    """A page render with the pipeline's predictions drawn on top.

    Only predictions get a box -- the handwriting itself is already visible in
    the render. Ground truth words the pipeline did not find are marked with an
    underline instead.
    """
    groups = []
    for row in page.rows:
        if row.status == "missed":
            x0, _, x1, y1 = row.gt_box  # type: ignore[misc]
            shapes = [
                f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"/>'
            ]
        else:
            x0, y0, x1, y1 = row.pred_box  # type: ignore[misc]
            shapes = [
                f'<rect x="{x0:.1f}" y="{y0:.1f}" '
                f'width="{x1 - x0:.1f}" height="{y1 - y0:.1f}"/>',
                f'<text x="{x0:.1f}" y="{y0 - 1.5:.1f}">'
                f"{html.escape(row.pred_text or '')}</text>",
            ]
        title = html.escape(f"GT: {row.gt_text or '—'} | pred: {row.pred_text or '—'}")
        shapes.insert(0, f"<title>{title}</title>")
        groups.append(f'<g class="{row.status}">{"".join(shapes)}</g>')

    return (
        f'<div class="page">'
        f'<img src="{page.image_uri}" alt="page render" loading="lazy">'
        f'<svg viewBox="0 0 {page.width:.1f} {page.height:.1f}" '
        f'preserveAspectRatio="none">{"".join(groups)}</svg></div>'
    )


def write_html_report(result: BenchmarkResult, out_path: Path) -> Path:
    """Write a self-contained HTML report showing every benchmark page.

    Each page of each benchmark sample is rendered once, with the ground truth
    boxes and the pipeline's predictions drawn on top.

    :param result: Result of `run_benchmark(..., collect_details=True)`.
    :param out_path: File to write the report to.
    :returns: `out_path`.
    """
    if result.details is None:
        raise ValueError(
            "No details collected -- run `run_benchmark(..., collect_details=True)`."
        )
    details = result.details

    body = [
        "<main>",
        "<h1>xournalpp_htr benchmark report</h1>",
        '<div class="meta">'
        f"Pipeline <code>{html.escape(details.pipeline_name)}</code> · "
        f"commit <code>{html.escape(details.git_sha)}</code> · "
        f"{html.escape(details.created_at)}<br>"
        f"precision {_pct(result.precision)} · recall {_pct(result.recall)} · "
        f"CER {_pct(result.cer)} ({_pct(result.cer_case_insensitive)} "
        f"case-insensitive) · "
        f"search quality {_pct(result.recall_times_accuracy)} · "
        f"exact match {_pct(result.word_accuracy)}"
        "</div>",
        _legend(),
    ]
    for sample in details.samples:
        body.append(f"<h2>{html.escape(sample.name)}</h2>")
        for page in sample.pages:
            body.append(f"<h3>Page {page.page_index + 1}</h3>")
            body.append(_page_figure(page))
    body.append("</main>")

    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Benchmark report — {html.escape(details.pipeline_name)}</title>"
        f"<style>{_REPORT_CSS}</style></head><body>" + "".join(body) + "</body></html>"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    return out_path
