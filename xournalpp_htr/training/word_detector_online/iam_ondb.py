"""IAM-OnDB forms as stroke pages with word-group labels.

IAM-OnDB gives the strokes per line, all in the coordinate frame of the form, and
only a transcription per line. A page is made of the lines of one form at their
original positions, so the detector sees line breaks.

Words are recovered like for SimpleHTR-OnDB, by cutting a line at its largest
horizontal gaps; lines whose gaps do not match the transcription are left out of
the page. The labels therefore follow a gap rule and cover only lines where it
works cleanly.

Optionally, synthetic non-text strokes (underlines, arrows, boxes, ...) are
added, as IAM-OnDB has none.
"""

from collections import Counter, defaultdict

import numpy as np

from xournalpp_htr.training.data.datasets import IAM_OnDB_Dataset
from xournalpp_htr.training.simple_htr_ondb.prepare_data import (
    form_of,
    list_line_ids,
    load_form_writers,
)
from xournalpp_htr.training.simple_htr_ondb.word_splitting import (
    group_tokens,
    split_into_words,
)
from xournalpp_htr.training.word_detector_online.data import NON_TEXT, StrokePage
from xournalpp_htr.training.word_detector_online.features import (
    bounding_boxes,
    page_scale,
)
from xournalpp_htr.training.word_detector_online.synthetic import (
    KINDS,
    Stroke,
    random_element,
)
from xournalpp_htr.xio import load_IAM_OnDB_dataset

#: Mean number of synthetic non-text elements added per page.
NON_TEXT_ELEMENTS_PER_PAGE = 2.0


def _squeeze_word_gaps(
    strokes: list[Stroke], groups: list[list[int]], factor: float
) -> list[Stroke]:
    """Move the words of a line together so each gap between words shrinks.

    Every gap between consecutive words (left to right) becomes `factor` times
    its size; the words themselves are untouched. The labels stay valid, the
    words just sit closer, like in tightly written text.
    """
    result = list(strokes)
    shift, previous_right = 0.0, None
    for group in groups:
        left = min(strokes[i][0].min() for i in group)
        right = max(strokes[i][0].max() for i in group)
        if previous_right is not None:
            shift += (left - previous_right) * (1.0 - factor)
        previous_right = right
        for i in group:
            result[i] = (strokes[i][0] - shift, strokes[i][1])
    return result


def _load_line(
    line_id: str,
    root,
    min_gap_ratio: float,
    gap_squeeze: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[Stroke] | None, np.ndarray | None, str | None]:
    """Return the strokes and word labels of a line, or why it is unusable.

    With `gap_squeeze` ``(low, high)``, the gaps between words shrink by a
    factor drawn uniformly from that range for the line.
    """
    df, text = IAM_OnDB_Dataset.load_IAM_OnDB_sample(line_id, root)
    df["y"] *= -1  # the loader flips y upwards; documents have y pointing down

    words, reason = group_tokens(text)
    if reason:
        return None, None, reason
    strokes = [
        (g["x"].to_numpy(), g["y"].to_numpy()) for _, g in df.groupby("stroke_nr")
    ]
    extents = np.array([[x.min(), x.max()] for x, _ in strokes])
    groups, reason = split_into_words(extents, len(words), min_gap_ratio)
    if reason:
        return None, None, reason

    if gap_squeeze is not None:
        strokes = _squeeze_word_gaps(strokes, groups, rng.uniform(*gap_squeeze))

    labels = np.empty(len(strokes), dtype=int)
    for word_id, group in enumerate(groups):
        labels[group] = word_id
    return strokes, labels, None


def _add_non_text(
    lines: list[tuple[list[Stroke], np.ndarray]], rng: np.random.Generator
) -> list[tuple[list[Stroke], np.ndarray]]:
    """Insert random non-text elements after randomly chosen lines.

    Elements are drawn after the line they sit next to, as in writing order.
    """
    result = [(list(strokes), labels) for strokes, labels in lines]
    for _ in range(rng.poisson(NON_TEXT_ELEMENTS_PER_PAGE)):
        k = int(rng.integers(len(lines)))
        strokes, labels = lines[k]
        boxes = bounding_boxes(_as_page(strokes, labels))
        line_box = (
            boxes[:, 0].min(),
            boxes[:, 1].min(),
            boxes[:, 2].max(),
            boxes[:, 3].max(),
        )
        element = random_element(
            str(rng.choice(KINDS)), line_box, page_scale(boxes), rng
        )
        extra_strokes, extra_labels = result[k]
        result[k] = (
            extra_strokes + element,
            np.concatenate([extra_labels, np.full(len(element), NON_TEXT)]),
        )
    return result


def _as_page(strokes: list[Stroke], labels: np.ndarray) -> StrokePage:
    return StrokePage("", 0, 0, strokes, labels)


def load_iam_ondb_pages(
    n_forms: int,
    seed: int = 0,
    min_gap_ratio: float = 1.5,
    synthetic_non_text: bool = True,
    gap_squeeze: tuple[float, float] | None = None,
    skip_forms: int = 0,
) -> tuple[list[StrokePage], Counter]:
    """Return `n_forms` random IAM-OnDB forms as pages, and why lines were rejected.

    The `document` of a page is its writer, so training can hold out writers.
    Forms without a single usable line are skipped. `gap_squeeze` shrinks the gaps
    between words, see `_load_line`, to teach the detector tighter word spacing.

    `skip_forms` leaves out the first forms of the shuffled order, so a second
    call with the same `seed` can draw forms the first one did not use.
    """
    root = load_IAM_OnDB_dataset()
    writers = load_form_writers(root)
    lines_of_form = defaultdict(list)
    for line_id in list_line_ids(root):
        lines_of_form[form_of(line_id)].append(line_id)

    rng = np.random.default_rng(seed)
    forms = sorted(lines_of_form)
    rng.shuffle(forms)
    forms = forms[skip_forms:]

    pages, stats = [], Counter()
    for form in forms:
        if len(pages) == n_forms:
            break
        lines = []
        for line_id in lines_of_form[form]:
            strokes, labels, reason = _load_line(
                line_id, root, min_gap_ratio, gap_squeeze, rng
            )
            if reason:
                stats[f"rejected_{reason}"] += 1
                continue
            offset = max((int(lab.max()) + 1 for _, lab in lines), default=0)
            lines.append((strokes, labels + offset))
            stats["kept_lines"] += 1
        if not lines:
            stats["skipped_forms"] += 1
            continue

        if synthetic_non_text:
            lines = _add_non_text(lines, rng)
        pages.append(
            StrokePage(
                document=f"iam:{writers[form]}",
                page_index=0,
                layer_index=0,
                strokes=[s for strokes, _ in lines for s in strokes],
                labels=np.concatenate([labels for _, labels in lines]),
            )
        )
    stats["pages"] = len(pages)
    stats["non_text_strokes"] = sum(int((p.labels == NON_TEXT).sum()) for p in pages)
    stats["strokes"] = sum(len(p.strokes) for p in pages)
    return pages, stats
