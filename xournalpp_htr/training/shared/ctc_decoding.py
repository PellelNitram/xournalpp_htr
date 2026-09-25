"""CTC beam search decoding via ``pyctcdecode``.

Base-deps-only (numpy, pyctcdecode). Shared between training-time evaluation
and inference-time decoding, mirroring ``postprocessing.py``. See ADR 006
section 3.
"""

import numpy as np
from pyctcdecode import build_ctcdecoder
from pyctcdecode.decoder import BeamSearchDecoderCTC


def build_beam_decoder(
    charset: list[str], unigrams: list[str] | None = None
) -> BeamSearchDecoderCTC:
    """Build a beam search decoder for ``charset``.

    Our CTC blank is the last class (index ``len(charset)``), matching
    ``network.greedy_decode`` and ``SimpleHTRModel.recognize``.
    ``pyctcdecode`` detects the blank position by looking for a known token
    (``""``, ``"<pad>"``, ...) in ``labels``; appending an explicit ``""``
    keeps the blank at the same last index without depending on its fallback
    behaviour.

    ``unigrams``, when given, biases the beam towards known words (a real
    lexicon) without needing a trained KenLM language model -- see
    ``build_vocabulary.py``. ``None`` (the default) is plain beam search.
    """
    labels = [*charset, ""]
    return build_ctcdecoder(labels=labels, unigrams=unigrams)


def beam_decode(log_probs: np.ndarray, decoder: BeamSearchDecoderCTC) -> str:
    """Beam search CTC decoding for a single sample.

    Args:
        log_probs: (seq_len, num_classes) natural-log probabilities.
        decoder: decoder built by :func:`build_beam_decoder`.

    Returns:
        Decoded string.
    """
    return decoder.decode(log_probs)
