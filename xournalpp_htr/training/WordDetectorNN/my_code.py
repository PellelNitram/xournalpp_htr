from typing import Optional
from typing import List, Tuple
import urllib.request
import json
from pathlib import Path
from typing import NamedTuple
import torch
from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn as nn
import torch.nn.functional as F
from git import Repo

# TODO: how to add w and h in all type annotations and datatype definitions?


class ImageDimensions(NamedTuple):
    height: int
    width: int

class BoundingBox:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, label: Optional[str] = None):
        """
        Initialize a bounding box.
        (x_min, y_min): top-left corner
        (x_max, y_max): bottom-right corner
        label: optional class label
        """
        # self.x_min: float = float(x_min)
        # self.y_min: float = float(y_min)
        # self.x_max: float = float(x_max)
        # self.y_max: float = float(y_max)
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.label: Optional[str] = label
        
    def translate(self, dx: float, dy: float) -> "BoundingBox":
        """Translate the bounding box by (dx, dy)."""
        bbox_translated = BoundingBox(
            self.x_min + dx,
            self.y_min + dy,
            self.x_max + dx,
            self.y_max + dy,
            self.label,
        )
        return bbox_translated

    def scale(self, sx: float, sy: float) -> "BoundingBox":
        """Scale the bounding box by sx and sy."""
        bbox_scaled = BoundingBox(
            self.x_min * sx,
            self.y_min * sy,
            self.x_max * sx,
            self.y_max * sy,
            self.label,
        )
        return bbox_scaled

    def as_type(self, new_type) -> "BoundingBox":
        # TODO: This invalidates the above `float` types. Needs new type def therefore.
        return BoundingBox(
            new_type(self.x_min),
            new_type(self.y_min),
            new_type(self.x_max),
            new_type(self.y_max),
            self.label,
        )

    def scale_around_center(self, sx, sy) -> "BoundingBox":
        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2
        return BoundingBox(
            x_min=center_x - sx * (center_x - self.x_min),
            y_min=center_y - sy * (center_y - self.y_min),
            x_max=center_x + sx * (self.x_max - center_x),
            y_max=center_y + sy * (self.y_max - center_y),
            label=self.label,
        )

    def clip(self, clip_aabb) -> "BoundingBox":
        return BoundingBox(
            x_min=min(max(self.x_min, clip_aabb.x_min), clip_aabb.x_max),
            y_min=min(max(self.y_min, clip_aabb.y_min), clip_aabb.y_max),
            x_max=max(min(self.x_max, clip_aabb.x_max), clip_aabb.x_min),
            y_max=max(min(self.y_max, clip_aabb.y_max), clip_aabb.y_min),
            label=self.label,
        )

    def area(self):
        """Return the area of the bounding box."""
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def enlarge_to_int_grid(self) -> "BoundingBox":
        return BoundingBox(
            x_min=np.floor(self.x_min),
            y_min=np.floor(self.y_min),
            x_max=np.ceil(self.x_max),
            y_max=np.ceil(self.y_max),
            label=self.label,
        )

    def enlarge(self, margin_x: float, margin_y: float) -> "BoundingBox":
        return BoundingBox(
            x_min=self.x_min-margin_x,
            x_max=self.x_max+margin_x,
            y_min=self.y_min-margin_y,
            y_max=self.y_max+margin_y,
        )

    # def intersect(self, other):
    #     """Return the intersection area with another bounding box."""
    #     x_min = max(self.x_min, other.x_min)
    #     y_min = max(self.y_min, other.y_min)
    #     x_max = min(self.x_max, other.x_max)
    #     y_max = min(self.y_max, other.y_max)
    #     if x_min < x_max and y_min < y_max:
    #         return (x_max - x_min) * (y_max - y_min)
    #     return 0.0

    # def iou(self, other):
    #     """Return the Intersection over Union (IoU) with another bounding box."""
    #     inter = self.intersect(other)
    #     union = self.area() + other.area() - inter
    #     if union == 0:
    #         return 0.0
    #     return inter / union

    def __repr__(self) -> str:
        return f"BoundingBox(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max}, label={self.label})"

import pickle
import xml.etree.ElementTree as ET
from typing import List, Tuple
from typing import TypedDict

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# TODO later: Replace print with logging.
# TODO: deal w/ height and width better & relate to x and y - do that once I know things are working

# TODO: Rename
class MapOrdering:
    """order of the maps encoding the aabbs around the words"""
    SEG_WORD = 0
    SEG_SURROUNDING = 1
    SEG_BACKGROUND = 2
    GEO_TOP = 3
    GEO_BOTTOM = 4
    GEO_LEFT = 5
    GEO_RIGHT = 6
    NUM_MAPS = 7

def encode(input_size: ImageDimensions, output_size: ImageDimensions, gt):
    f = output_size.height / input_size.height
    gt_map = np.zeros((MapOrdering.NUM_MAPS,) + output_size)
    for aabb in gt:
        aabb = aabb.scale(f, f)

        # segmentation map
        aabb_clip = BoundingBox(0, 0, output_size.width - 1, output_size.height - 1)

        aabb_word = aabb.scale_around_center(0.5, 0.5).as_type(int).clip(aabb_clip)
        aabb_sur = aabb.as_type(int).clip(aabb_clip)
        # TODO: fix hack to get ints
        gt_map[MapOrdering.SEG_SURROUNDING, int(aabb_sur.y_min):int(aabb_sur.y_max) + 1, int(aabb_sur.x_min):int(aabb_sur.x_max) + 1] = 1
        gt_map[MapOrdering.SEG_SURROUNDING, int(aabb_word.y_min):int(aabb_word.y_max) + 1, int(aabb_word.x_min):int(aabb_word.x_max) + 1] = 0
        gt_map[MapOrdering.SEG_WORD, int(aabb_word.y_min):int(aabb_word.y_max) + 1, int(aabb_word.x_min):int(aabb_word.x_max) + 1] = 1

        # geometry map TODO vectorize
        for x in range(int(aabb_word.x_min), int(aabb_word.x_max) + 1):
            for y in range(int(aabb_word.y_min), int(aabb_word.y_max) + 1):
                gt_map[MapOrdering.GEO_TOP, y, x] = y - aabb.y_min
                gt_map[MapOrdering.GEO_BOTTOM, y, x] = aabb.y_max - y
                gt_map[MapOrdering.GEO_LEFT, y, x] = x - aabb.x_min
                gt_map[MapOrdering.GEO_RIGHT, y, x] = aabb.x_max - x

    gt_map[MapOrdering.SEG_BACKGROUND] = np.clip(
        1
        - gt_map[MapOrdering.SEG_WORD]
        - gt_map[MapOrdering.SEG_SURROUNDING],
        0,
        1
    )

    return gt_map

def subsample(idx, max_num):
    """restrict fg indices to a maximum number"""
    f = len(idx[0]) / max_num
    if f > 1:
        a = np.asarray([idx[0][int(j * f)] for j in range(max_num)], np.int64)
        b = np.asarray([idx[1][int(j * f)] for j in range(max_num)], np.int64)
        idx = (a, b)
    return idx

def fg_by_threshold(thres, max_num=None):
    """all pixels above threshold are fg pixels, optionally limited to a maximum number"""

    def func(seg_map):
        idx = np.where(seg_map > thres)
        if max_num is not None:
            idx = subsample(idx, max_num)
        return idx

    return func

def fg_by_cc(thres, max_num):
    """take a maximum number of pixels per connected component, but at least 3 (->DBSCAN minPts)"""

    def func(seg_map):
        seg_mask = (seg_map > thres).astype(np.uint8)
        num_labels, label_img = cv2.connectedComponents(seg_mask, connectivity=4)
        max_num_per_cc = max(max_num // (num_labels + 1), 3)  # at least 3 because of DBSCAN clustering

        all_idx = [np.empty(0, np.int64), np.empty(0, np.int64)]
        for curr_label in range(1, num_labels):
            curr_idx = np.where(label_img == curr_label)
            curr_idx = subsample(curr_idx, max_num_per_cc)
            all_idx[0] = np.append(all_idx[0], curr_idx[0])
            all_idx[1] = np.append(all_idx[1], curr_idx[1])
        return tuple(all_idx)

    return func

def decode(nn_prediction, scale=1.0, comp_fg=fg_by_threshold(0.5)) -> List[BoundingBox]:
    idx = comp_fg(nn_prediction[MapOrdering.SEG_WORD])
    nn_prediction_masked = nn_prediction[..., idx[0], idx[1]]
    bounding_boxes = []
    for yc, xc, pred in zip(idx[0], idx[1], nn_prediction_masked.T):
        t = pred[MapOrdering.GEO_TOP]
        b = pred[MapOrdering.GEO_BOTTOM]
        l = pred[MapOrdering.GEO_LEFT]
        r = pred[MapOrdering.GEO_RIGHT]
        bbox = BoundingBox(
            x_min=xc - l,
            x_max=xc + r,
            y_min=yc - t,
            y_max=yc + b
        )
        bounding_boxes.append(bbox.scale(scale, scale))
    return bounding_boxes

class IAM_Dataset_Element(TypedDict):
    image: np.ndarray
    bounding_boxes: List[BoundingBox]
    filename: str
    gt_encoded: np.ndarray

class IAM_Dataset(Dataset):
    """
    Loads, pre-processes, and caches the IAM Handwriting Database.

    This class handles the entire data preparation pipeline. On the first run, it
    processes all images and ground truth files, resizes them, and saves them
    to a cache file for extremely fast loading on subsequent runs.

    Inherits from `torch.utils.data.Dataset`, making it fully compatible with
    PyTorch's DataLoader.
    """

    # TODO: Open question: what is (x,y) direction and (w, h)? -> happy for now but need to investigate and make it more explicit for production model! heuristically, it seems like w -> x and h -> y.

    # TODO: Has potential to be reworked in one IAM_Ds and one based on top of that to transform to dataset used in this modeling approach and therefore DataLoader; can be done later once I know things work

    _GT_DIR_NAME = 'gt'
    _IMG_DIR_NAME = 'img'
    _IMG_EXT = '.png'
    _GT_EXT = '*.xml'

    def __init__(
        self,
        root_dir: Path,
        input_size: ImageDimensions,
        output_size: ImageDimensions,
        force_rebuild_cache: bool = False,
        transform = None,
        cache_path: Path = Path('dataset_cache.pickle'),
    ):
        """
        Initializes the dataset. Checks for a cache file first. If it doesn't
        exist, it builds one.

        Args:
            root_dir (Path): The root directory of the dataset, containing 'gt' and 'img' subdirectories.
            input_size (Tuple[int, int]): The target (height, width) for the network input images.
            loaded_img_scale (float): A factor to initially scale down images to reduce memory
                                      usage during pre-processing. Default is 0.25.
        """
        super().__init__()
        self.root_dir = root_dir
        self.input_size = input_size
        self.output_size = output_size
        self.input_width = input_size.width
        self.input_height = input_size.height
        self.output_width = output_size.width
        self.output_height = output_size.height
        self.transform = transform

        assert self.output_width / self.input_width == self.output_height / self.input_height, 'Input and output need to have same aspect ratio' # Same aspect ratio

        self.img_cache: List[np.ndarray] = []
        self.gt_cache: List[List[BoundingBox]] = []
        self.filename_cache: List[str] = []

        if cache_path.exists() and not force_rebuild_cache:
            print(f"Loading cached data from {cache_path}...")
            self._load_from_cache(cache_path)
        else:
            print(f"Cache not found. Building and caching data from {self.root_dir}...")
            self._preprocess_and_cache(cache_path)

    def _load_from_cache(self, cache_path: Path):
        """Loads pre-processed data from a pickle file."""
        with open(cache_path, 'rb') as f:
            self.img_cache, self.gt_cache, self.filename_cache = pickle.load(f)

    def _preprocess_and_cache(self, cache_path: Path):
        """Finds, processes, and caches all data samples."""
        gt_dir = self.root_dir / self._GT_DIR_NAME
        img_dir = self.root_dir / self._IMG_DIR_NAME

        fn_gts = sorted(gt_dir.glob(self._GT_EXT))
        print(f"Found {len(fn_gts)} ground truth files. Processing...")

        # TODO: Make this task parallel!

        for fn_gt in tqdm(fn_gts, desc="Preprocessing IAM Dataset"):
            fn_img = img_dir / (fn_gt.stem + self._IMG_EXT)
            if not fn_img.exists():
                continue

            # Load image and GT
            img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
            gt = self._parse_gt(fn_gt)

            # Pre-processing pipeline
            img, gt = self._crop_page_to_content(img, gt)
            img, gt = self._adjust_to_input_size(img, gt)

            self.img_cache.append(img)
            self.gt_cache.append(gt)
            self.filename_cache.append(fn_gt.stem)

        print(f"Preprocessing complete. Saving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump([self.img_cache, self.gt_cache, self.filename_cache], f)
        print("Cache saved successfully.")


    def _parse_gt(self, fn_gt: Path) -> List[BoundingBox]:
        """Parses an XML ground truth file to get word bounding boxes."""
        tree = ET.parse(fn_gt)
        root = tree.getroot()
        aabbs = []

        for line in root.findall("./handwritten-part/line"):
            for word in line.findall('./word'):
                x_min, x_max, y_min, y_max = float('inf'), 0, float('inf'), 0
                components = word.findall('./cmp')
                if not components:
                    continue

                for cmp in components:
                    x = float(cmp.attrib['x'])
                    y = float(cmp.attrib['y'])
                    w = float(cmp.attrib['width'])
                    h = float(cmp.attrib['height'])
                    x_min = min(x_min, x)
                    x_max = max(x_max, x + w)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y + h)
                
                text = word.attrib['text']
                
                # Scale coordinates to match the initially scaled image
                aabb = BoundingBox(x_min, y_min, x_max, y_max, text)
                aabbs.append(aabb)
        return aabbs

    def _crop_page_to_content(self, img: np.ndarray, gt: List[BoundingBox]) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Crops the image to the bounding box containing all words."""
        x_min = min(aabb.x_min for aabb in gt)
        x_max = max(aabb.x_max for aabb in gt)
        y_min = min(aabb.y_min for aabb in gt)
        y_max = max(aabb.y_max for aabb in gt)

        gt_crop = [aabb.translate(-x_min, -y_min) for aabb in gt]
        img_crop = img[int(y_min):int(y_max), int(x_min):int(x_max)] # TODO: Round correctly as opposed to just int'ing
        return img_crop, gt_crop

    def _adjust_to_input_size(self, img: np.ndarray, gt: List[BoundingBox]) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Resizes the image and AABBs to the final network input size."""
        h, w = img.shape
        # print(f'incoming image: {h=} {w=}') # TODO: Use logging here.
        sx = self.input_width / w
        sy = self.input_height / h
        # print(f'scaling factors: {sx=} {sy=}') # TODO: Use logging here.
        gt_resized = [aabb.scale(sx, sy) for aabb in gt]
        img_resized = cv2.resize(img, dsize=(self.input_width, self.input_height)) # cv2 uses (w, h)
        return img_resized, gt_resized

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.img_cache)

    def __getitem__(self, idx: int) -> IAM_Dataset_Element:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing:
            - The pre-processed image as a NumPy array.
            - A list of AABB objects for the ground truth words.
        """
        image = self.img_cache[idx]
        bounding_boxes = self.gt_cache[idx]
        if self.transform:
            # print('[INFO] transformation applied')
            image, bounding_boxes = self.transform(image, bounding_boxes)
        gt_encoded = encode(self.input_size, self.output_size, bounding_boxes)
        return {
            'image': image,
            'bounding_boxes': bounding_boxes,
            'filename': self.filename_cache[idx],
            'gt_encoded': gt_encoded,
        }
    
    def store_element_as_image(
            self,
            idx: int,
            output_path: Path,
            draw_bboxes: bool = False,
            store_gt_encoded: bool = False,
        ) -> List[Path]:
        """
        Saves a dataset element as an image with bounding boxes drawn on it.
        
        Args:
            idx (int): The index of the dataset element to save.
            output_path (Path): The path where the image should be saved.
        """
        # Get the element
        element = self[idx]
        img = element['image'].copy()  # Copy to avoid modifying the cached image
        bboxes = element['bounding_boxes']
        
        # Convert grayscale to BGR for colored bounding boxes
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes
        if draw_bboxes:
            for bbox in bboxes:
                # Convert float coordinates to integers
                x_min = int(bbox.x_min)
                y_min = int(bbox.y_min)
                x_max = int(bbox.x_max)
                y_max = int(bbox.y_max)
                
                # Draw rectangle (green color, thickness=2)
                cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Save the image
        cv2.imwrite(str(output_path), img_color)
        files_to_return = [output_path]

        # TODO: Add text there

        if store_gt_encoded:
            gt_encoded = element['gt_encoded']
            for key, value in MapOrdering.__dict__.items():
                if '__' not in key and key != 'NUM_MAPS':
                    data = gt_encoded[value].copy()
                    data_normalised = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8) # Required for storing as image

                    name = Path(output_path.stem + f'__{key.lower()}' + output_path.suffix)
                    files_to_return.append(name)

                    # Convert grayscale to BGR for colored bounding boxes
                    img_color = cv2.cvtColor(data_normalised, cv2.COLOR_GRAY2BGR)
                    
                    # Save the image
                    cv2.imwrite(str(name), img_color)

        return files_to_return

def dummy_transform(img, aabbs):
    return img, aabbs

from my_code import IAM_Dataset_Element
from typing import List, Dict
from typing import TypedDict
from my_code import BoundingBox


class Dataloader_Element(TypedDict):
    images: torch.tensor
    bounding_boxes: List[List[BoundingBox]]
    gt_encoded: torch.tensor

def custom_collate_fn(batch: List[IAM_Dataset_Element]) -> Dataloader_Element:
    """
    Custom collate function to handle IAM_Dataset_Element batches.
    """
    
    batch_images = []
    batch_gt_encodeds = []
    batch_bounding_boxes = []

    for sample in batch:
        image = sample['image']
        gt_encoded = sample['gt_encoded']
        bounding_boxes = sample['bounding_boxes']
        batch_images.append(image[None, ...].astype(np.float32))
        batch_gt_encodeds.append(gt_encoded.astype(np.float32))
        batch_bounding_boxes.append(bounding_boxes)

    batch_images = np.stack(batch_images, axis=0)
    batch_gt_encodeds = np.stack(batch_gt_encodeds, axis=0)

    batch_images = torch.from_numpy(batch_images)
    batch_gt_encodeds = torch.from_numpy(batch_gt_encodeds)
    
    return {
        'images': batch_images,
        'gt_encoded': batch_gt_encodeds,
        'bounding_boxes': batch_bounding_boxes
    }

def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN


def compute_iou(ra: BoundingBox, rb: BoundingBox) -> float:
    """intersection over union of two axis aligned rectangles ra and rb"""
    if ra.x_max < rb.x_min or rb.x_max < ra.x_min or ra.y_max < rb.y_min or rb.y_max < ra.y_min:
        return 0

    l = max(ra.x_min, rb.x_min)
    r = min(ra.x_max, rb.x_max)
    t = max(ra.y_min, rb.y_min)
    b = min(ra.y_max, rb.y_max)

    intersection = (r - l) * (b - t)
    union = ra.area() + rb.area() - intersection

    iou = intersection / union
    return iou

def compute_dist_mat(aabbs: List[BoundingBox]) -> np.ndarray:
    """Jaccard distance matrix of all pairs of aabbs"""
    num_aabbs = len(aabbs)

    dists = np.zeros((num_aabbs, num_aabbs))
    for i in range(num_aabbs):
        for j in range(num_aabbs):
            if j > i:
                break

            dists[i, j] = dists[j, i] = 1 - compute_iou(aabbs[i], aabbs[j])

    return dists

def cluster_aabbs(aabbs: List[BoundingBox]) -> List[BoundingBox]:
    """cluster aabbs using DBSCAN and the Jaccard distance between bounding boxes"""
    if len(aabbs) < 2:
        return aabbs

    dists = compute_dist_mat(aabbs)
    clustering = DBSCAN(eps=0.7, min_samples=3, metric='precomputed').fit(dists)

    clusters = defaultdict(list)
    for i, c in enumerate(clustering.labels_):
        if c == -1:
            continue
        clusters[c].append(aabbs[i])

    res_aabbs = []
    for curr_cluster in clusters.values():
        xmin = np.median([aabb.x_min for aabb in curr_cluster])
        xmax = np.median([aabb.x_max for aabb in curr_cluster])
        ymin = np.median([aabb.y_min for aabb in curr_cluster])
        ymax = np.median([aabb.y_max for aabb in curr_cluster])
        res_aabbs.append(BoundingBox(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax))

    return res_aabbs

def compute_dist_mat_2(aabbs1: List[BoundingBox], aabbs2: List[BoundingBox]) -> np.ndarray:
    """Jaccard distance matrix of all pairs of aabbs from lists aabbs1 and aabbs2"""
    num_aabbs1 = len(aabbs1)
    num_aabbs2 = len(aabbs2)

    dists = np.zeros((num_aabbs1, num_aabbs2))
    for i in range(num_aabbs1):
        for j in range(num_aabbs2):
            dists[i, j] = 1 - compute_iou(aabbs1[i], aabbs2[j])

    return dists

def binary_classification_metrics(gt_aabbs: List[BoundingBox], pred_aabbs: List[BoundingBox]) -> dict:
    iou_thres = 0.7

    ious = 1 - compute_dist_mat_2(gt_aabbs, pred_aabbs)
    match_counter = (ious > iou_thres).astype(int)
    gt_counter = np.sum(match_counter, axis=1)
    pred_counter = np.sum(match_counter, axis=0)

    tp = np.count_nonzero(pred_counter == 1)
    fp = np.count_nonzero(pred_counter == 0)
    fn = np.count_nonzero(gt_counter == 0)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }

def draw_bboxes_on_image(
    img: np.ndarray,
    aabbs: List[BoundingBox],
    denormalise: bool=True,
) -> np.ndarray:
    """
    Draws bounding boxes on an image.

    Args:
        img (np.ndarray): The image on which to draw the bounding boxes.
        aabbs (List[BoundingBox]): List of bounding boxes to draw.

    Returns:
        np.ndarray: The image with drawn bounding boxes.
    """
    img = img.copy()
    if denormalise:
        img = ((img + 0.5) * 255).astype(np.uint8) # Reverse normalization
    is_grayscale = len(img.shape) == 2 # Otherwise the image is interpreted as BGR b/c we use cv2
    if is_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for aabb in aabbs:
        aabb = aabb.enlarge_to_int_grid().as_type(int) # TODO: as_type doesn't work, grr

        cv2.rectangle(
            img,
            (
                int(aabb.x_min),
                int(aabb.y_min),
            ),
            (
                int(aabb.x_max),
                int(aabb.y_max),
            ),
            (0, 0, 255), # Red
            2
        )

    return img

def compute_loss(y, gt_map):

    # TODO: refactoring possibility: compute area, can be used 2 to 3 times
    # TODO: add weights in loss and hyperparameter-tune them

    # 1. segmentation loss
    target_labels = torch.argmax(gt_map[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1], dim=1)
    loss_seg = F.cross_entropy(y[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1], target_labels)

    # 2. geometry loss
    # distances to all sides of aabb
    t = torch.minimum(y[:, MapOrdering.GEO_TOP], gt_map[:, MapOrdering.GEO_TOP])
    b = torch.minimum(y[:, MapOrdering.GEO_BOTTOM], gt_map[:, MapOrdering.GEO_BOTTOM])
    l = torch.minimum(y[:, MapOrdering.GEO_LEFT], gt_map[:, MapOrdering.GEO_LEFT])
    r = torch.minimum(y[:, MapOrdering.GEO_RIGHT], gt_map[:, MapOrdering.GEO_RIGHT])

    # area of predicted aabb
    y_width = y[:, MapOrdering.GEO_LEFT, ...] + y[:, MapOrdering.GEO_RIGHT, ...]
    y_height = y[:, MapOrdering.GEO_TOP, ...] + y[:, MapOrdering.GEO_BOTTOM, ...]
    area1 = y_width * y_height

    # area of gt aabb
    gt_width = gt_map[:, MapOrdering.GEO_LEFT, ...] + gt_map[:, MapOrdering.GEO_RIGHT, ...]
    gt_height = gt_map[:, MapOrdering.GEO_TOP, ...] + gt_map[:, MapOrdering.GEO_BOTTOM, ...]
    area2 = gt_width * gt_height

    # compute intersection over union
    intersection = (r + l) * (b + t)
    union = area1 + area2 - intersection
    eps = 0.01  # avoid division by 0
    iou = intersection / (union + eps)
    iou = iou[gt_map[:, MapOrdering.SEG_WORD] > 0]
    loss_aabb = -torch.log(torch.mean(iou))

    # total loss is simply the sum of both losses
    loss = loss_seg + loss_aabb
    return loss

# If you were using Bottleneck for other ResNet versions:
# from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class ModifiedResNet18(ResNet):
    def __init__(self, **kwargs):
        # Initialize with BasicBlock and standard ResNet-18 layers
        # num_classes is irrelevant here as we won't use the fc layer
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=1000, **kwargs)

        # 1. Modify the first convolutional layer for 1-channel (grayscale) input
        # Original resnet.conv1 is Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # We need Conv2d(1, 64, ...)
        original_conv1 = self.conv1
        self.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )  # bias is False in original ResNet conv1

        # Optional: If you wanted to initialize weights similarly to torchvision:
        # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        # However, if you load custom pretrained weights for the whole model later,
        # this specific initialization might be overwritten.

        # We don't need the final fully connected layer for feature extraction
        del self.fc
        # self.avgpool is also not strictly needed for the U-Net style features,
        # but it doesn't hurt to keep it if not used. You could 'del self.avgpool' too.

    def _forward_impl(self, x: torch.Tensor):
        # This is largely copied from torchvision.models.resnet.ResNet._forward_impl
        # but modified to return intermediate features.

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu(x)  # Corresponds to bb1 in WordDetectorNet (before maxpool)
        x = self.maxpool(out1)

        out2 = self.layer1(x)  # Corresponds to bb2
        out3 = self.layer2(out2)  # Corresponds to bb3
        out4 = self.layer3(out3)  # Corresponds to bb4
        out5 = self.layer4(out4)  # Corresponds to bb5

        # WordDetectorNet expects (bb5, bb4, bb3, bb2, bb1)
        return out5, out4, out3, out2, out1

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)


def compute_scale_down(input_size, output_size):
    """compute scale down factor of neural network, given input and output size"""
    return output_size[0] / input_size[0]


class UpscaleAndConcatLayer(torch.nn.Module):
    """
    take small map with cx channels
    upscale to size of large map (s*s)
    concat large map with cy channels and upscaled small map
    apply conv and output map with cz channels
    """

    def __init__(self, cx, cy, cz):
        super(UpscaleAndConcatLayer, self).__init__()
        self.conv = torch.nn.Conv2d(cx + cy, cz, 3, padding=1)

    def forward(self, x, y, s):
        x = F.interpolate(x, s)
        z = torch.cat((x, y), 1)
        z = F.relu(self.conv(z))
        return z


class WordDetectorNet(torch.nn.Module):
    input_size = (448, 448)
    output_size = (224, 224)
    # v-- TODO: It's a hack to keep both. I do so now b/c I don't know the order (also, doesn't matter b/c same values)
    input_size_ImageDimensions: ImageDimensions = ImageDimensions(width=448, height=448)
    output_size_ImageDimensions: ImageDimensions = ImageDimensions(width=224, height=224)
    scale_down = compute_scale_down(input_size, output_size)

    def __init__(self):
        super(WordDetectorNet, self).__init__()

        # Use the modified ResNet18 for feature extraction
        self.backbone = ModifiedResNet18()
        # All weights in the backbone will be randomly initialized.

        self.up1 = UpscaleAndConcatLayer(512, 256, 256)  # input//16
        self.up2 = UpscaleAndConcatLayer(256, 128, 128)  # input//8
        self.up3 = UpscaleAndConcatLayer(128, 64, 64)  # input//4
        self.up4 = UpscaleAndConcatLayer(64, 64, 32)  # input//2

        self.conv1 = torch.nn.Conv2d(32, MapOrdering.NUM_MAPS, 3, 1, padding=1)

    @staticmethod
    def scale_shape(s, f):
        assert s[0] % f == 0 and s[1] % f == 0
        return s[0] // f, s[1] // f

    def output_activation(self, x, apply_softmax):
        if apply_softmax:
            seg = torch.softmax(
                x[:, MapOrdering.SEG_WORD : MapOrdering.SEG_BACKGROUND + 1], dim=1
            )
        else:
            seg = x[:, MapOrdering.SEG_WORD : MapOrdering.SEG_BACKGROUND + 1]
        geo = torch.sigmoid(x[:, MapOrdering.GEO_TOP :]) * self.input_size[0] # TODO: Understand this
        y = torch.cat([seg, geo], dim=1)
        return y

    def forward(self, x, apply_softmax=False):
        s = x.shape[2:]  # Original image shape HxW
        bb5, bb4, bb3, bb2, bb1 = self.backbone(x)

        y = self.up1(bb5, bb4, self.scale_shape(s, 16))
        # up2 takes y (H/16, 256ch) and bb3 (H/8, 128ch). Upscales y to H/8. Output: H/8, 128ch.
        y = self.up2(y, bb3, self.scale_shape(s, 8))
        # up3 takes y (H/8, 128ch) and bb2 (H/4, 64ch). Upscales y to H/4. Output: H/4, 64ch.
        y = self.up3(y, bb2, self.scale_shape(s, 4))
        # up4 takes y (H/4, 64ch) and bb1 (H/2, 64ch). Upscales y to H/2. Output: H/2, 32ch.
        y = self.up4(y, bb1, self.scale_shape(s, 2))

        y = self.conv1(
            y
        )  # Final convolution to get NUM_MAPS channels. Output: H/2, NUM_MAPS ch.

        return self.output_activation(y, apply_softmax)

# ==========
# Transforms
# ==========

def normalize_image_transform(image, bounding_boxes):
    image_new = (image / 255.0) - 0.5
    return image_new, bounding_boxes

# =========
# Inference
# =========

def run_image_through_network(
        image_grayscale: np.ndarray,
        model_path: Path=Path('best_model.pth'),
        device: str='cuda',
    ) -> List[BoundingBox]:


    # Load model
    # ==========

    model = WordDetectorNet()  # instantiate your model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ==============
    # Pre processing
    # ==============

    image_gray_rescaled = cv2.resize(image_grayscale, WordDetectorNet.input_size)

    image_grayscale_transformed, _ = normalize_image_transform(image_gray_rescaled, None) # Only works w/ current transformation setup

    image_grayscale_transformed = image_grayscale_transformed.astype(np.float32)
    
    image_grayscale_transformed = torch.from_numpy(image_grayscale_transformed[None, None, :, :]).to(device)

    # =========
    # Inference
    # =========

    with torch.no_grad():
        output_image = model(image_grayscale_transformed, apply_softmax=True)

    assert output_image[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND+1, :, :].min() >= 0.0
    assert output_image[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND+1, :, :].max() <= 1.0

    output_image = output_image.to('cpu').numpy()

    output_image = output_image[0, :, :, :]

    # ===============
    # Post processing
    # ===============

    decoded_aabbs = decode(
        output_image,
        scale=WordDetectorNet.input_size[0] / WordDetectorNet.output_size[0],
        comp_fg=fg_by_cc(thres=0.5, max_num=1000),
    )
    model_input_image = image_grayscale_transformed[0, 0, :, :].to('cpu').numpy()
    h, w = model_input_image.shape
    aabbs = [aabb.clip(BoundingBox(0, 0, w - 1, h - 1)) for aabb in decoded_aabbs]  # bounding box must be inside input img
    clustered_aabbs = cluster_aabbs(aabbs)

    return {
        'aabbs': clustered_aabbs,
        'model_input_image': model_input_image,
    }

class CustomEncoder(json.JSONEncoder):
    """This is to make non-standard items serialisable for `json.dump(s)`."""
    def default(self, obj):
        if isinstance(obj, Path): # Store `Path` objects
            return str(obj)
        return super().default(obj)

def get_git_commit_hash(repo_path: Path = Path("."), short: bool = False) -> str:
    """
    Get the current Git commit hash using GitPython.
    
    :param repo_path: Path to the Git repository (default: current directory).
    :param short: Whether to return the short hash.
    :return: Commit hash string, or None if unavailable.
    """
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        commit_hash: str = repo.head.commit.hexsha
        return commit_hash[:7] if short else commit_hash
    except Exception as e:
        print(f"Error while getting git commit hash from {repo_path}: {e}")
        return '-1'

def url_exists(url: str) -> bool:
    """Check if a URL exists using a HEAD request."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception as e:
        print(e)
        return False

def get_example_list() -> List[List]:
    """Only return examples that still exist online."""
    links_to_images = [
        'https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/cvl.jpg',
        'https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/random.jpg',
        'https://raw.githubusercontent.com/githubharald/WordDetectorNN/master/data/test/bentham.jpg',
    ]
    result = []
    for link in links_to_images:
        image_exists = url_exists(link)
        if image_exists:
            result.append([link, 0])
    return result