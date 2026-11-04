"""IAM-OnDB forms as stroke pages with word-group labels.

IAM-OnDB gives the strokes per line, all in the coordinate frame of the form, and
only a transcription per line. A page is made of the lines of one form at their
original positions, so the detector sees line breaks.

Words are recovered like for SimpleHTR-OnDB, by cutting a line at its largest
horizontal gaps; lines whose gaps do not match the transcription are left out of
the page. The labels therefore follow a gap rule and cover only lines where it
works cleanly.

The gaps between words can be shrunk, as IAM-OnDB words are spaced more widely
than in tightly written notes.
"""

from collections import Counter, defaultdict

import numpy as np

from xournalpp_htr.training.simple_htr_ondb.word_splitting import (
    group_tokens,
    split_into_words,
)
from xournalpp_htr.training.word_detector_online.strokes import StrokePage
from xournalpp_htr.xio import load_IAM_OnDB_dataset

# The IAM-OnDB dataset helpers need torch (the training extra), so they are imported
# inside the functions that read the dataset; this module stays importable without it.

Stroke = tuple[np.ndarray, np.ndarray]

#: IAM-OnDB forms the published detector is trained on.
TRAINING_FORMS = 700

#: Range of the factor the gaps between words are shrunk by when training it.
TRAINING_GAP_SQUEEZE = (0.3, 1.0)


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
    from xournalpp_htr.training.data.datasets import IAM_OnDB_Dataset

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


def load_iam_ondb_pages(
    n_forms: int,
    seed: int = 0,
    min_gap_ratio: float = 1.5,
    gap_squeeze: tuple[float, float] | None = None,
) -> tuple[list[StrokePage], Counter]:
    """Return `n_forms` random IAM-OnDB forms as pages, and why lines were rejected.

    The `document` of a page is its writer, so training can hold out writers.
    Forms without a single usable line are skipped. `gap_squeeze` shrinks the
    gaps between words, see `_load_line`.
    """
    from xournalpp_htr.training.simple_htr_ondb.prepare_data import (
        form_of,
        list_line_ids,
        load_form_writers,
    )

    root = load_IAM_OnDB_dataset()
    writers = load_form_writers(root)
    lines_of_form = defaultdict(list)
    for line_id in list_line_ids(root):
        lines_of_form[form_of(line_id)].append(line_id)

    rng = np.random.default_rng(seed)
    forms = sorted(lines_of_form)
    rng.shuffle(forms)

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
    stats["strokes"] = sum(len(p.strokes) for p in pages)
    return pages, stats
