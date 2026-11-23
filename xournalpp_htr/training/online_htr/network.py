"""Bidirectional LSTM + CTC network of Carbune et al. (2020).

Port of OnlineHTR's ``LitModule1`` to plain PyTorch. Differences to the
reference:

* The CTC blank is the *last* class (like SimpleHTR), so the charset indexes
  directly and ``shared.ctc_decoding`` applies unchanged.
* Padded batches are packed, so the backward LSTM does not start on padding.
* ``zero_infinity`` in the CTC loss: strokes resampled to few points can be
  shorter than their label.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from xournalpp_htr.training.online_htr.config import ModelConfig


class OnlineHTRNet(nn.Module):
    """Features ``(T, N, C_in)`` to log-probabilities ``(T, N, num_classes)``."""

    def __init__(self, num_classes: int, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_channels,
            hidden_size=cfg.nodes_per_layer,
            num_layers=cfg.number_of_layers,
            dropout=cfg.dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * cfg.nodes_per_layer, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the network.

        :param x: Padded features, shape ``(T, N, C_in)``.
        :param lengths: True length of each sequence, shape ``(N,)``. ``None``
            means no padding (e.g. one sample, as in the ONNX export).
        """
        if lengths is None:
            out, _ = self.lstm(x)
        else:
            packed = pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, total_length=x.shape[0])
        return F.log_softmax(self.linear(out), dim=2)


def compute_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )


def greedy_decode(
    log_probs: torch.Tensor,
    charset: list[str],
    lengths: torch.Tensor | None = None,
) -> list[str]:
    """Best-path CTC decoding of ``(T, N, num_classes)`` log-probabilities."""
    blank = len(charset)
    best = log_probs.argmax(dim=2).T.cpu().numpy()  # (N, T)
    results = []
    for i, path in enumerate(best):
        if lengths is not None:
            path = path[: int(lengths[i])]
        chars, previous = [], blank
        for index in path:
            if index != previous and index != blank:
                chars.append(charset[index])
            previous = index
        results.append("".join(chars))
    return results
