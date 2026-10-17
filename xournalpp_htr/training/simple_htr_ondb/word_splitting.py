"""Split IAM-OnDB lines into words: transcription tokens and stroke gaps.

Pure numpy, so it is importable (and testable) in the base install.

IAM-OnDB only gives a transcription per *line*. To get word-level samples, the
line is cut at its largest horizontal gaps. A line is only usable if exactly as
many clear gaps exist as the transcription has word boundaries.
"""

import numpy as np

# Standalone tokens in the transcription ("Mr ." / "( foo )") that stick to the
# previous / next word when written by hand.
_ATTACH_TO_PREVIOUS = set(".,;:!?)")
_ATTACH_TO_NEXT = set("(")


def group_tokens(text: str) -> tuple[list[str], str | None]:
    """Turn a transcription line into words, gluing punctuation tokens on.

    :param text: Transcription line; tokens are separated by spaces.
    :returns: ``(words, reject_reason)``. ``reject_reason`` is ``None`` for
        usable lines. Lines with standalone punctuation whose position is
        ambiguous (quotes, dashes, ...) are rejected rather than guessed.
    """
    words: list[str] = []
    prefix = ""
    for token in text.split():
        chars = set(token)
        if chars <= _ATTACH_TO_PREVIOUS:
            if not words:
                return [], "leading_punctuation"
            words[-1] += token
        elif chars <= _ATTACH_TO_NEXT:
            prefix += token
        elif not any(c.isalnum() for c in token):
            return [], "ambiguous_punctuation"
        else:
            words.append(prefix + token)
            prefix = ""
    if prefix:
        return [], "dangling_opening_punctuation"
    if not words:
        return [], "empty_line"
    return words, None


def split_into_words(
    extents: np.ndarray, n_words: int, min_gap_ratio: float = 1.5
) -> tuple[list[list[int]] | None, str | None]:
    """Group strokes into ``n_words`` words by cutting at the largest gaps.

    Strokes whose horizontal extents overlap (i-dots, t-bars, connected
    letters) are merged first. The ``n_words - 1`` largest gaps between the
    resulting blobs are word boundaries. If more blobs than words remain, the
    smallest boundary gap must exceed the largest non-boundary gap by
    ``min_gap_ratio``, otherwise the split is ambiguous.

    :param extents: Array of shape ``(n_strokes, 2)`` with ``(x_min, x_max)``.
    :param n_words: Number of words the transcription has.
    :param min_gap_ratio: Required separation between boundary and
        non-boundary gaps.
    :returns: ``(groups, reject_reason)``; ``groups`` is a list of stroke index
        lists in left-to-right order, or ``None`` if the line is rejected.
    """
    if len(extents) == 0:
        return None, "no_strokes"

    # Merge overlapping strokes into blobs, left to right.
    order = np.argsort(extents[:, 0], kind="stable")
    blobs: list[dict] = []
    for i in order:
        x_min, x_max = extents[i]
        if blobs and x_min <= blobs[-1]["x_max"]:
            blobs[-1]["x_max"] = max(blobs[-1]["x_max"], x_max)
            blobs[-1]["indices"].append(int(i))
        else:
            blobs.append({"x_min": x_min, "x_max": x_max, "indices": [int(i)]})

    if len(blobs) < n_words:
        return None, "too_few_blobs"
    if n_words == 1:
        return [[i for b in blobs for i in b["indices"]]], None

    gaps = np.array(
        [blobs[k + 1]["x_min"] - blobs[k]["x_max"] for k in range(len(blobs) - 1)]
    )
    by_size = np.argsort(-gaps, kind="stable")
    boundaries = sorted(by_size[: n_words - 1].tolist())

    if len(gaps) > n_words - 1:
        smallest_boundary = gaps[by_size[n_words - 2]]
        largest_other = gaps[by_size[n_words - 1]]
        if smallest_boundary < min_gap_ratio * largest_other:
            return None, "ambiguous_gaps"

    groups: list[list[int]] = []
    start = 0
    for cut in boundaries + [len(blobs) - 1]:
        groups.append([i for b in blobs[start : cut + 1] for i in b["indices"]])
        start = cut + 1
    return groups, None
