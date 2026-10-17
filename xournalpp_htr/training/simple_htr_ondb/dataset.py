"""Rendered IAM-OnDB word dataset for SimpleHTR-OnDB training.

Reads the word crops and ``manifest.jsonl`` written by ``prepare_data``. The
train/val split is fixed by writer in the manifest. Copy of
``simple_htr/dataset.py`` with the IAM word-file loading replaced.
Requires the ``training-simple-htr-ondb`` extra (torch).
"""

import json
from pathlib import Path
from typing import List, TypedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def build_charset(texts: List[str]) -> List[str]:
    return sorted({c for t in texts for c in t})


def encode_text(text: str, char_to_idx: dict) -> List[int]:
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode_indices(indices: List[int], idx_to_char: dict) -> str:
    return "".join(idx_to_char[i] for i in indices if i in idx_to_char)


def preprocess_image(
    img: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
    y_off = (target_height - new_h) // 2
    x_off = (target_width - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def _apply_augmentation(img: np.ndarray) -> np.ndarray:
    if np.random.random() < 0.25:
        kernel_size = np.random.choice([3, 5, 7])
        if img.dtype != np.uint8:
            img_uint8 = ((img + 0.5) * 255).clip(0, 255).astype(np.uint8)
            img_uint8 = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
            img = img_uint8.astype(np.float32) / 255.0 - 0.5
        else:
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    if np.random.random() < 0.5:
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-6:
            img = (img - img_min) / (img_max - img_min) - 0.5
        factor = np.random.uniform(0.5, 1.0)
        img = img * factor

    if np.random.random() < 0.25:
        noise = np.random.uniform(-0.05, 0.05, size=img.shape).astype(np.float32)
        img = img + noise

    if np.random.random() < 0.2:
        kernel = np.ones((3, 3), np.uint8)
        if img.dtype != np.uint8:
            img_uint8 = ((img + 0.5) * 255).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = img
        if np.random.random() < 0.5:
            img_uint8 = cv2.erode(img_uint8, kernel, iterations=1)
        else:
            img_uint8 = cv2.dilate(img_uint8, kernel, iterations=1)
        if img.dtype != np.uint8:
            img = img_uint8.astype(np.float32) / 255.0 - 0.5
        else:
            img = img_uint8

    if np.random.random() < 0.5:
        h, w = img.shape[:2]
        if img.dtype != np.uint8:
            img_uint8 = ((img + 0.5) * 255).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = img
        scale = np.random.uniform(0.75, 1.05)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img_uint8, (new_w, new_h))
        # Random position on white canvas
        canvas = np.ones((h, w), dtype=np.uint8) * 255
        paste_h, paste_w = min(new_h, h), min(new_w, w)
        max_y = max(h - paste_h, 0)
        max_x = max(w - paste_w, 0)
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)
        canvas[y : y + paste_h, x : x + paste_w] = resized[:paste_h, :paste_w]
        if img.dtype != np.uint8:
            img = canvas.astype(np.float32) / 255.0 - 0.5
        else:
            img = canvas

    return img


class OnDB_Words_Element(TypedDict):
    image: np.ndarray
    text: str
    encoded: List[int]
    filename: str


class OnDB_Words_Dataset(Dataset):
    """Preprocesses one split of the rendered IAM-OnDB crops and keeps it in memory.

    :param data_dir: Output directory of ``prepare_data``.
    :param split: ``"train"`` or ``"val"``.
    :param charset: Characters to encode. Built from this split if ``None``;
        pass the train charset for ``val``. Characters outside it still count
        as errors in validation, they are just not encodable as targets.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        target_height: int,
        target_width: int,
        charset: List[str] | None = None,
        augment: bool = False,
    ):
        super().__init__()
        self.augment = augment

        rows = []
        with open(data_dir / "manifest.jsonl") as f:
            for line in f:
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
        if not rows:
            raise ValueError(f"No samples for split {split!r} in {data_dir}.")

        self.img_cache: List[np.ndarray] = []
        self.text_cache: List[str] = []
        self.filename_cache: List[str] = []
        for row in tqdm(rows, desc=f"Preprocessing {split}"):
            img = cv2.imread(str(data_dir / row["image"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(data_dir / row["image"])
            self.img_cache.append(preprocess_image(img, target_height, target_width))
            self.text_cache.append(row["text"])
            self.filename_cache.append(row["image"])

        self.charset = (
            charset if charset is not None else build_charset(self.text_cache)
        )
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.idx_to_char = dict(enumerate(self.charset))

    def __len__(self) -> int:
        return len(self.img_cache)

    def __getitem__(self, idx: int) -> OnDB_Words_Element:
        image = self.img_cache[idx].copy()
        text = self.text_cache[idx]

        image = image.astype(np.float32) / 255.0 - 0.5

        if self.augment:
            image = _apply_augmentation(image)

        encoded = encode_text(text, self.char_to_idx)
        return {
            "image": image,
            "text": text,
            "encoded": encoded,
            "filename": self.filename_cache[idx],
        }


class Dataloader_Element(TypedDict):
    images: torch.Tensor
    texts: List[str]
    encoded: List[List[int]]
    target_lengths: torch.Tensor
    targets: torch.Tensor


def custom_collate_fn(batch: List[OnDB_Words_Element]) -> Dataloader_Element:
    """Collate ``OnDB_Words_Element`` batches for the ``DataLoader``."""
    images = torch.from_numpy(
        np.stack([s["image"][None, ...] for s in batch], axis=0).astype(np.float32)
    )
    texts = [s["text"] for s in batch]
    encoded = [s["encoded"] for s in batch]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = torch.tensor([idx for e in encoded for idx in e], dtype=torch.long)
    return {
        "images": images,
        "texts": texts,
        "encoded": encoded,
        "target_lengths": target_lengths,
        "targets": targets,
    }
