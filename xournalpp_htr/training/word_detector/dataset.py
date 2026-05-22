"""IAM dataset loading and ground-truth encoding for WordDetector training.

Requires the ``training-word-detector`` extra (torch).
"""

import logging
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, TypedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from xournalpp_htr.training.shared.bounding_box import BoundingBox, ImageDimensions
from xournalpp_htr.training.shared.postprocessing import MapOrdering

logger = logging.getLogger(__name__)

CACHE_VERSION = "v2"


def encode(input_size: ImageDimensions, output_size: ImageDimensions, gt):
    f = output_size.height / input_size.height
    gt_map = np.zeros((MapOrdering.NUM_MAPS,) + output_size)
    for aabb in gt:
        aabb = aabb.scale(f, f)

        aabb_clip = BoundingBox(0, 0, output_size.width - 1, output_size.height - 1)

        aabb_word = aabb.scale_around_center(0.5, 0.5).as_type(int).clip(aabb_clip)
        aabb_sur = aabb.as_type(int).clip(aabb_clip)
        gt_map[
            MapOrdering.SEG_SURROUNDING,
            int(aabb_sur.y_min) : int(aabb_sur.y_max) + 1,
            int(aabb_sur.x_min) : int(aabb_sur.x_max) + 1,
        ] = 1
        gt_map[
            MapOrdering.SEG_SURROUNDING,
            int(aabb_word.y_min) : int(aabb_word.y_max) + 1,
            int(aabb_word.x_min) : int(aabb_word.x_max) + 1,
        ] = 0
        gt_map[
            MapOrdering.SEG_WORD,
            int(aabb_word.y_min) : int(aabb_word.y_max) + 1,
            int(aabb_word.x_min) : int(aabb_word.x_max) + 1,
        ] = 1

        y_min_w, y_max_w = int(aabb_word.y_min), int(aabb_word.y_max) + 1
        x_min_w, x_max_w = int(aabb_word.x_min), int(aabb_word.x_max) + 1
        ys = np.arange(y_min_w, y_max_w)[:, None]
        xs = np.arange(x_min_w, x_max_w)[None, :]
        gt_map[MapOrdering.GEO_TOP, y_min_w:y_max_w, x_min_w:x_max_w] = ys - aabb.y_min
        gt_map[MapOrdering.GEO_BOTTOM, y_min_w:y_max_w, x_min_w:x_max_w] = (
            aabb.y_max - ys
        )
        gt_map[MapOrdering.GEO_LEFT, y_min_w:y_max_w, x_min_w:x_max_w] = xs - aabb.x_min
        gt_map[MapOrdering.GEO_RIGHT, y_min_w:y_max_w, x_min_w:x_max_w] = (
            aabb.x_max - xs
        )

    gt_map[MapOrdering.SEG_BACKGROUND] = np.clip(
        1 - gt_map[MapOrdering.SEG_WORD] - gt_map[MapOrdering.SEG_SURROUNDING], 0, 1
    )
    return gt_map


class IAM_Dataset_Element(TypedDict):
    image: np.ndarray
    bounding_boxes: List[BoundingBox]
    filename: str
    gt_encoded: np.ndarray


class IAM_Dataset(Dataset):
    """Loads, pre-processes and caches the IAM Handwriting Database.

    On first run it processes all images and ground truth files, resizes them
    and saves a pickle cache for fast subsequent loading. Compatible with
    PyTorch's ``DataLoader``.
    """

    _GT_DIR_NAME = "xml"
    _IMG_DIR_NAME = "forms"
    _IMG_EXT = ".png"
    _GT_EXT = "*.xml"

    def __init__(
        self,
        root_dir: Path,
        input_size: ImageDimensions,
        output_size: ImageDimensions,
        force_rebuild_cache: bool = False,
        transform=None,
        cache_path: Path = Path("dataset_cache.pickle"),
    ):
        super().__init__()
        self.root_dir = root_dir
        self.input_size = input_size
        self.output_size = output_size
        self.input_width = input_size.width
        self.input_height = input_size.height
        self.output_width = output_size.width
        self.output_height = output_size.height
        self.transform = transform

        assert (
            self.output_width / self.input_width
            == self.output_height / self.input_height
        ), "Input and output need to have same aspect ratio"

        self.img_cache: List[np.ndarray] = []
        self.gt_cache: List[List[BoundingBox]] = []
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
            self.img_cache, self.gt_cache, self.filename_cache = payload["data"]
            return True
        return False

    def _preprocess_and_cache(self, cache_path: Path):
        gt_dir = self.root_dir / self._GT_DIR_NAME
        img_dir = self.root_dir / self._IMG_DIR_NAME

        fn_gts = sorted(gt_dir.glob(self._GT_EXT))
        print(f"Found {len(fn_gts)} ground truth files. Processing...")

        for fn_gt in tqdm(fn_gts, desc="Preprocessing IAM Dataset"):
            fn_img = img_dir / (fn_gt.stem + self._IMG_EXT)
            if not fn_img.exists():
                continue

            img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
            gt = self._parse_gt(fn_gt)

            img, gt = self._crop_page_to_content(img, gt)
            img, gt = self._adjust_to_input_size(img, gt)

            self.img_cache.append(img)
            self.gt_cache.append(gt)
            self.filename_cache.append(fn_gt.stem)

        print(f"Preprocessing complete. Saving cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "version": CACHE_VERSION,
                    "data": [self.img_cache, self.gt_cache, self.filename_cache],
                },
                f,
            )
        print("Cache saved successfully.")

    def _parse_gt(self, fn_gt: Path) -> List[BoundingBox]:
        tree = ET.parse(fn_gt)
        root = tree.getroot()
        aabbs = []

        for line in root.findall("./handwritten-part/line"):
            for word in line.findall("./word"):
                x_min, x_max, y_min, y_max = float("inf"), 0, float("inf"), 0
                components = word.findall("./cmp")
                if not components:
                    continue

                for cmp in components:
                    x = float(cmp.attrib["x"])
                    y = float(cmp.attrib["y"])
                    w = float(cmp.attrib["width"])
                    h = float(cmp.attrib["height"])
                    x_min = min(x_min, x)
                    x_max = max(x_max, x + w)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y + h)

                text = word.attrib["text"]
                aabbs.append(BoundingBox(x_min, y_min, x_max, y_max, text))
        return aabbs

    def _crop_page_to_content(
        self, img: np.ndarray, gt: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        x_min = min(aabb.x_min for aabb in gt)
        x_max = max(aabb.x_max for aabb in gt)
        y_min = min(aabb.y_min for aabb in gt)
        y_max = max(aabb.y_max for aabb in gt)

        gt_crop = [aabb.translate(-x_min, -y_min) for aabb in gt]
        img_crop = img[int(y_min) : int(y_max), int(x_min) : int(x_max)]
        return img_crop, gt_crop

    def _adjust_to_input_size(
        self, img: np.ndarray, gt: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        h, w = img.shape
        sx = self.input_width / w
        sy = self.input_height / h
        gt_resized = [aabb.scale(sx, sy) for aabb in gt]
        img_resized = cv2.resize(img, dsize=(self.input_width, self.input_height))
        return img_resized, gt_resized

    def __len__(self) -> int:
        return len(self.img_cache)

    def __getitem__(self, idx: int) -> IAM_Dataset_Element:
        image = self.img_cache[idx]
        bounding_boxes = self.gt_cache[idx]
        if self.transform:
            image, bounding_boxes = self.transform(image, bounding_boxes)
        gt_encoded = encode(self.input_size, self.output_size, bounding_boxes)
        return {
            "image": image,
            "bounding_boxes": bounding_boxes,
            "filename": self.filename_cache[idx],
            "gt_encoded": gt_encoded,
        }


def dummy_transform(img, aabbs):
    return img, aabbs


class Dataloader_Element(TypedDict):
    images: torch.Tensor
    bounding_boxes: List[List[BoundingBox]]
    gt_encoded: torch.Tensor


def custom_collate_fn(batch: List[IAM_Dataset_Element]) -> Dataloader_Element:
    """Collate ``IAM_Dataset_Element`` batches for the ``DataLoader``."""
    batch_images = []
    batch_gt_encodeds = []
    batch_bounding_boxes = []

    for sample in batch:
        batch_images.append(sample["image"][None, ...].astype(np.float32))
        batch_gt_encodeds.append(sample["gt_encoded"].astype(np.float32))
        batch_bounding_boxes.append(sample["bounding_boxes"])

    batch_images = torch.from_numpy(np.stack(batch_images, axis=0))
    batch_gt_encodeds = torch.from_numpy(np.stack(batch_gt_encodeds, axis=0))

    return {
        "images": batch_images,
        "gt_encoded": batch_gt_encodeds,
        "bounding_boxes": batch_bounding_boxes,
    }
