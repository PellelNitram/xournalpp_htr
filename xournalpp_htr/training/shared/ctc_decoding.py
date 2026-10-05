"""CTC beam search decoding via ``pyctcdecode``.

Base-deps-only (numpy, pyctcdecode). Shared between training-time evaluation
and inference-time decoding, mirroring ``postprocessing.py``. See ADR 006
section 3.
"""

import numpy as np
from pyctcdecode import build_ctcdecoder
from pyctcdecode.decoder import BeamSearchDecoderCTC


def build_beam_decoder(charset: list[str]) -> BeamSearchDecoderCTC:
    """Build a plain (no language model) beam search decoder for ``charset``.

    Our CTC blank is the last class (index ``len(charset)``), matching
    ``network.greedy_decode`` and ``SimpleHTRModel.recognize``.
    ``pyctcdecode`` detects the blank position by looking for a known token
    (``""``, ``"<pad>"``, ...) in ``labels``; appending an explicit ``""``
    keeps the blank at the same last index without depending on its fallback
    behaviour.

    Deliberately does not take a ``unigrams``/lexicon argument: pyctcdecode's
    own lexicon/LM scoring assumes multi-word output separated by a space
    token to find word boundaries, but SimpleHTR decodes one word per crop
    with no space in its charset -- that machinery is a silent no-op here
    ("Space token ' ' missing from vocabulary."). See :func:`beam_decode`'s
    ``vocabulary`` argument for the lexicon bias we implement instead.
    """
    labels = [*charset, ""]
    return build_ctcdecoder(labels=labels)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def beam_decode(
    log_probs: np.ndarray,
    decoder: BeamSearchDecoderCTC,
    vocabulary: set[str] | None = None,
    beam_width: int = 500,
    max_correction_distance: int = 4,
) -> str:
    """Beam search CTC decoding for a single sample.

    Args:
        log_probs: (seq_len, num_classes) natural-log probabilities.
        decoder: decoder built by :func:`build_beam_decoder`.
        vocabulary: lower-cased word set (see ``build_vocabulary.py``). If
            given, picks the beam candidate that is a real word *and* closest
            (edit distance) to the top beam, capped at
            ``max_correction_distance``, falling back to the top beam if none
            qualify. Taking the first real word in beam-score order instead
            (an earlier version of this function) let an unrelated-but-higher
            -scoring real word outrank the correct one; distance-to-top-beam
            favours small corrections over big jumps to an equally plausible
            wrong word. ``None`` (the default) just takes the top beam.
        beam_width: beam width to search when ``vocabulary`` is given
            (ignored otherwise, where pyctcdecode's own default applies).
            Matches ``analyze_beam_coverage.py``'s default so results are
            comparable to its reachability numbers.
        max_correction_distance: reject vocabulary matches further than this
            from the top beam, so a real-but-wildly-different word is never
            preferred over the network's own best guess.

    Returns:
        Decoded string.
    """
    if vocabulary is None:
        return decoder.decode(log_probs)

    beams = decoder.decode_beams(log_probs, beam_width=beam_width)
    top_text = beams[0][0]

    best_text, best_distance = None, max_correction_distance + 1
    for beam in beams:
        text = beam[0]
        if text.lower() not in vocabulary:
            continue
        distance = _edit_distance(text, top_text)
        if distance < best_distance:
            best_text, best_distance = text, distance
            if distance == 0:
                break

    return best_text if best_text is not None else top_text
