"""SimpleHTR network architecture and CTC loss.

Reimplementation of `Harald Scheidl's SimpleHTR
<https://github.com/githubharald/SimpleHTR>`_: 5 CNN layers, 2 bidirectional
LSTM layers, and a CTC output layer. Takes a grayscale word image and produces
character-level logits for CTC decoding.

Requires the ``training-simple-htr`` extra (torch).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from xournalpp_htr.training.simple_htr.config import ModelConfig


class SimpleHTRNet(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.num_classes = num_classes
        channels = cfg.cnn_channels

        self.cnn = nn.Sequential(
            # Layer 1: 1 -> 32, pool 2x2
            nn.Conv2d(1, channels[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Layer 2: 32 -> 64, pool 2x2
            nn.Conv2d(channels[0], channels[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Layer 3: 64 -> 128, pool height only (2x1)
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # Layer 4: 128 -> 128, pool height only (2x1)
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # Layer 5: 128 -> 256, pool height only (2x1)
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )

        # After CNN: height = 32 / (2*2*2*2*2) = 1, width = 128 / (2*2) = 32
        cnn_out_height = cfg.input_height // 32
        rnn_input_size = channels[4] * cnn_out_height

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(cfg.rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 1, height, width) grayscale image tensor.

        Returns:
            (seq_len, batch, num_classes) log-probabilities for CTC.
        """
        features = self.cnn(x)  # (batch, channels, h', w')
        batch, channels, h, w = features.shape
        # Collapse height into channels, keep width as sequence dimension
        features = features.permute(0, 3, 1, 2)  # (batch, w', channels, h')
        features = features.reshape(batch, w, channels * h)  # (batch, w', channels*h')
        rnn_out, _ = self.rnn(features)  # (batch, w', 2*hidden)
        logits = self.fc(rnn_out)  # (batch, w', num_classes)
        # CTC expects (seq_len, batch, num_classes)
        return F.log_softmax(logits.permute(1, 0, 2), dim=2)


def compute_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
) -> torch.Tensor:
    input_lengths = torch.full(
        (log_probs.size(1),), log_probs.size(0), dtype=torch.long
    )
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=blank)


def greedy_decode(log_probs: torch.Tensor, charset: List[str]) -> list[str]:
    """Best-path CTC decoding.

    Args:
        log_probs: (seq_len, batch, num_classes) from ``forward()``.
        charset: list of characters (blank index = len(charset)).

    Returns:
        List of decoded strings, one per batch element.
    """
    blank = len(charset)
    predictions = log_probs.argmax(dim=2).permute(1, 0)  # (batch, seq_len)
    results = []
    for pred in predictions:
        chars = []
        prev = blank
        for idx in pred:
            idx = idx.item()
            if idx != prev and idx != blank:
                chars.append(charset[idx])
            prev = idx
        results.append("".join(chars))
    return results
