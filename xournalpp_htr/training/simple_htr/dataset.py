"""IAM word-level dataset loading for SimpleHTR training.

Loads pre-cropped word images from the IAM Handwriting Database
(``data/words/`` directory) with transcriptions from ``data/ascii/words.txt``.
Requires the ``training-simple-htr`` extra (torch).
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, TypedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

CACHE_VERSION = "v1"

CHARSET = list(
    " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = dict(enumerate(CHARSET))


def encode_text(text: str) -> List[int]:
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_indices(indices: List[int]) -> str:
    return "".join(IDX_TO_CHAR[i] for i in indices if i in IDX_TO_CHAR)


def preprocess_image(
    img: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = min(int(w * scale), target_width)
    resized = cv2.resize(img, (new_w, target_height))

    padded = np.ones((target_height, target_width), dtype=np.uint8) * 255
    padded[:, :new_w] = resized
    return padded


def _apply_augmentation(img: np.ndarray) -> np.ndarray:
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
        kernel = np.ones((2, 2), np.uint8)
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

    return img


class IAM_Words_Element(TypedDict):
    image: np.ndarray
    text: str
    encoded: List[int]
    filename: str


class IAM_Words_Dataset(Dataset):
    """Loads, pre-processes and caches the IAM word-level dataset.

    Reads word images from ``data/words/`` and ground-truth transcriptions
    from ``data/ascii/words.txt``. Only words with segmentation status ``ok``
    are included. Compatible with PyTorch's ``DataLoader``.
    """

    def __init__(
        self,
        root_dir: Path,
        target_height: int,
        target_width: int,
        force_rebuild_cache: bool = False,
        augment: bool = False,
        cache_path: Path = Path("dataset_cache.pickle"),
    ):
        super().__init__()
        self.root_dir = root_dir
        self.target_height = target_height
        self.target_width = target_width
        self.augment = augment

        self.img_cache: List[np.ndarray] = []
        self.text_cache: List[str] = []
        self.filename_cache: List[str] = []

        if cache_path.exists() and not force_rebuild_cache:
            if self._load_from_cache(cache_path):
                return
            logger.warning("Cache version mismatch, rebuilding.")
        print(f"Building and caching data from {self.root_dir}...")
        self._preprocess_and_cache(cache_path)

    def _load_from_cache(self, cache_path: Path) -> bool:
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and payload.get("version") == CACHE_VERSION:
            self.img_cache, self.text_cache, self.filename_cache = payload["data"]
            return True
        return False

    def _preprocess_and_cache(self, cache_path: Path):
        words_file = self.root_dir / "ascii" / "words.txt"
        words_dir = self.root_dir / "words"

        entries = self._parse_words_file(words_file)
        print(f"Found {len(entries)} word entries. Processing...")

        for word_id, text in tqdm(entries, desc="Preprocessing IAM Words"):
            parts = word_id.split("-")
            img_path = (
                words_dir / parts[0] / f"{parts[0]}-{parts[1]}" / f"{word_id}.png"
            )
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = preprocess_image(img, self.target_height, self.target_width)
            self.img_cache.append(img)
            self.text_cache.append(text)
            self.filename_cache.append(word_id)

        print(f"Preprocessing complete. {len(self.img_cache)} samples loaded.")
        print(f"Saving cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "version": CACHE_VERSION,
                    "data": [
                        self.img_cache,
                        self.text_cache,
                        self.filename_cache,
                    ],
                },
                f,
            )
        print("Cache saved successfully.")

    @staticmethod
    def _parse_words_file(words_file: Path) -> List[Tuple[str, str]]:
        entries = []
        with open(words_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                seg_status = parts[1]
                if seg_status != "ok":
                    continue
                word_id = parts[0]
                text = parts[-1]
                # words.txt uses | for space within a word (rare)
                text = text.replace("|", " ")
                if all(c in CHAR_TO_IDX for c in text):
                    entries.append((word_id, text))
        return entries

    def __len__(self) -> int:
        return len(self.img_cache)

    def __getitem__(self, idx: int) -> IAM_Words_Element:
        image = self.img_cache[idx].copy()
        text = self.text_cache[idx]

        image = image.astype(np.float32) / 255.0 - 0.5

        if self.augment:
            image = _apply_augmentation(image)

        encoded = encode_text(text)
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


def custom_collate_fn(batch: List[IAM_Words_Element]) -> Dataloader_Element:
    """Collate ``IAM_Words_Element`` batches for the ``DataLoader``."""
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
