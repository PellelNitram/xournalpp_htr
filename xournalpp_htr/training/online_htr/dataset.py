"""IAM-OnDB samples (lines and words) for OnlineHTR training.

IAM-OnDB only transcribes whole lines. Besides lines, this also yields words:
a line is cut at its largest stroke gaps, as for SimpleHTR-OnDB, which works
for lines whose gaps match the transcription. The pipeline recognises words,
so these are the samples closest to inference.

Line labels glue standalone punctuation onto the previous word (``Mr .`` ->
``Mr.``), like the word labels, so both agree on how text is written.
Train/val are split by writer.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from xournalpp_htr.training.online_htr.config import AugmentationConfig, DataConfig
from xournalpp_htr.training.online_htr.features import Stroke, strokes_to_features
from xournalpp_htr.training.simple_htr_ondb.word_splitting import (
    group_tokens,
    split_into_words,
)


@dataclass
class Sample:
    strokes: list[Stroke]  # y pointing down, writing order
    text: str
    writer: str
    source: str  # "line" or "word"
    name: str


def load_form_writers(root: Path) -> dict[str, str]:
    """Map form id (``a01-000u``) to writer id from ``forms.txt``."""
    mapping = {}
    for line in (root / "forms.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            form, writer, *_ = line.split()
            mapping[form] = writer
    return mapping


def assign_splits(writers: list[str], val_fraction: float, seed: int) -> dict[str, str]:
    """Assign whole writers to ``train`` or ``val``."""
    unique = sorted(set(writers))
    np.random.default_rng(seed).shuffle(unique)
    n_val = max(1, round(len(unique) * val_fraction))
    return {w: ("val" if i < n_val else "train") for i, w in enumerate(unique)}


def _samples_of_line(entry: dict, writer: str, cfg: DataConfig) -> list[Sample]:
    name, text = entry["sample_name"], entry["label"]
    stroke_nr = np.asarray(entry["stroke_nr"])
    x, y = np.asarray(entry["x"]), -np.asarray(entry["y"])  # loader flips y upwards
    strokes = [(x[stroke_nr == k], y[stroke_nr == k]) for k in np.unique(stroke_nr)]
    if not strokes:
        return []

    words, reason = group_tokens(text)
    samples = []
    if cfg.level in ("line", "both"):
        label = text if reason else " ".join(words)
        samples.append(Sample(strokes, label, writer, "line", name))

    if cfg.level in ("word", "both") and not reason:
        extents = np.array([[sx.min(), sx.max()] for sx, _ in strokes])
        groups, split_reason = split_into_words(extents, len(words), cfg.min_gap_ratio)
        if not split_reason:
            for k, (word, group) in enumerate(zip(words, groups, strict=True)):
                samples.append(
                    Sample(
                        [strokes[i] for i in sorted(group)],
                        word,
                        writer,
                        "word",
                        f"{name}_{k:02d}",
                    )
                )
    return samples


def load_samples(root: Path, cfg: DataConfig) -> dict[str, list[Sample]]:
    """Load IAM-OnDB from ``root`` as ``{"train": ..., "val": ...}`` samples."""
    from xournalpp_htr.training.data.datasets import IAM_OnDB_Dataset

    entries = IAM_OnDB_Dataset(root, limit=cfg.limit).data
    writers = load_form_writers(root)
    split_of = assign_splits(
        list(writers.values()), cfg.val_writer_fraction, cfg.split_seed
    )

    result: dict[str, list[Sample]] = {"train": [], "val": []}
    skipped = []
    for entry in entries:
        writer = writers.get(entry["sample_name"].rsplit("-", 1)[0])
        if writer is None:  # a form missing from forms.txt: no writer to split by
            skipped.append(entry["sample_name"])
            continue
        result[split_of[writer]].extend(_samples_of_line(entry, writer, cfg))
    if skipped:
        print(f"Skipped {len(skipped)} lines without a writer: {skipped}")
    return result


def build_charset(samples: list[Sample]) -> list[str]:
    return sorted({c for s in samples for c in s.text})


def augment_strokes(
    strokes: list[Stroke], cfg: AugmentationConfig, rng: np.random.Generator
) -> list[Stroke]:
    """Random x-scale, shear and rotation of the whole ink."""
    angle = math.radians(rng.uniform(-cfg.rotation, cfg.rotation))
    shear = rng.uniform(-cfg.shear, cfg.shear)
    x_scale = rng.uniform(1 - cfg.x_scale, 1 + cfg.x_scale)
    cos, sin = math.cos(angle), math.sin(angle)
    result = []
    for x, y in strokes:
        x = (x + shear * y) * x_scale
        result.append((cos * x - sin * y, sin * x + cos * y))
    return result


class OnlineHTRDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        charset: list[str],
        points_per_unit: float,
        augment: AugmentationConfig | None = None,
    ):
        self.samples = samples
        self.charset = charset
        self.index = {c: i for i, c in enumerate(charset)}
        self.points_per_unit = points_per_unit
        self.augment = augment if augment is not None and augment.enabled else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        sample = self.samples[i]
        strokes = sample.strokes
        if self.augment is not None:
            strokes = augment_strokes(strokes, self.augment, np.random.default_rng())
        ink = strokes_to_features(strokes, self.points_per_unit)
        # Characters unseen in training are dropped from the target; the CER
        # still counts them via `text`.
        target = [self.index[c] for c in sample.text if c in self.index]
        return {
            "ink": torch.from_numpy(ink),
            "target": torch.tensor(target, dtype=torch.long),
            "text": sample.text,
            "source": sample.source,
        }


def custom_collate_fn(batch: list[dict]) -> dict:
    return {
        "ink": pad_sequence([b["ink"] for b in batch]),  # (T, N, C)
        "ink_lengths": torch.tensor([len(b["ink"]) for b in batch]),
        "targets": torch.cat([b["target"] for b in batch]),
        "target_lengths": torch.tensor([len(b["target"]) for b in batch]),
        "texts": [b["text"] for b in batch],
        "sources": [b["source"] for b in batch],
    }
