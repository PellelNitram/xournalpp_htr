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


def beam_decode(
    log_probs: np.ndarray,
    decoder: BeamSearchDecoderCTC,
    vocabulary: set[str] | None = None,
    beam_width: int = 500,
) -> str:
    """Beam search CTC decoding for a single sample.

    Args:
        log_probs: (seq_len, num_classes) natural-log probabilities.
        decoder: decoder built by :func:`build_beam_decoder`.
        vocabulary: lower-cased word set (see ``build_vocabulary.py``). If
            given, returns the beam's first candidate that is an exact
            (case-insensitive) match, falling back to the top-scoring beam if
            none match. ``None`` (the default) just takes the top beam.
        beam_width: beam width to search when ``vocabulary`` is given
            (ignored otherwise, where pyctcdecode's own default applies).
            Matches ``analyze_beam_coverage.py``'s default so results are
            comparable to its reachability numbers.

    Returns:
        Decoded string.
    """
    if vocabulary is None:
        return decoder.decode(log_probs)

    beams = decoder.decode_beams(log_probs, beam_width=beam_width)
    for beam in beams:
        text = beam[0]
        if text.lower() in vocabulary:
            return text
    return beams[0][0]
