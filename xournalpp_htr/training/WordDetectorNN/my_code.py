from typing import Optional
from pathlib import Path
from typing import NamedTuple


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

    # def area(self):
    #     """Return the area of the bounding box."""
    #     return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

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
    print('IMPLEMENT ME!!!!')
    # for aabb in gt:
    #     aabb = aabb.scale(f, f)

    #     # segmentation map
    #     aabb_clip = BoundingBox(0, 0, output_size[0] - 1, output_size[1] - 1)

    #     aabb_word = aabb.scale_around_center(0.5, 0.5).as_type(int).clip(aabb_clip)
    #     aabb_sur = aabb.as_type(int).clip(aabb_clip)
    #     # TODO: fix hack to get ints
    #     gt_map[MapOrdering.SEG_SURROUNDING, int(aabb_sur.y_min):int(aabb_sur.y_max) + 1, int(aabb_sur.x_min):int(aabb_sur.x_max) + 1] = 1
    #     gt_map[MapOrdering.SEG_SURROUNDING, int(aabb_word.y_min):int(aabb_word.y_max) + 1, int(aabb_word.x_min):int(aabb_word.x_max) + 1] = 0
    #     gt_map[MapOrdering.SEG_WORD, int(aabb_word.y_min):int(aabb_word.y_max) + 1, int(aabb_word.x_min):int(aabb_word.x_max) + 1] = 1

    #     # geometry map TODO vectorize
    #     for x in range(int(aabb_word.x_min), int(aabb_word.x_max) + 1):
    #         for y in range(int(aabb_word.y_min), int(aabb_word.y_max) + 1):
    #             gt_map[MapOrdering.GEO_TOP, y, x] = y - aabb.y_min
    #             gt_map[MapOrdering.GEO_BOTTOM, y, x] = aabb.y_max - y
    #             gt_map[MapOrdering.GEO_LEFT, y, x] = x - aabb.x_min
    #             gt_map[MapOrdering.GEO_RIGHT, y, x] = aabb.x_max - x

    # gt_map[MapOrdering.SEG_BACKGROUND] = np.clip(
    #     1
    #     - gt_map[MapOrdering.SEG_WORD]
    #     - gt_map[MapOrdering.SEG_SURROUNDING],
    #     0,
    #     1
    # )

    return gt_map

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
    _CACHE_FILENAME = 'dataset_cache.pickle'
    _IMG_EXT = '.png'
    _GT_EXT = '*.xml'

    def __init__(
        self,
        root_dir: Path,
        input_size: ImageDimensions,
        output_size: ImageDimensions,
        force_rebuild_cache: bool = False,
        transform = None,
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

        cache_path = self.root_dir / self._CACHE_FILENAME
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
            print('[INFO] transformation applied')
            image, bounding_boxes = self.transform(image, bounding_boxes)
        gt_encoded = encode(self.input_size, self.output_size, bounding_boxes)
        return {
            'image': image,
            'bounding_boxes': bounding_boxes,
            'filename': self.filename_cache[idx],
            'gt_encoded': gt_encoded,
        }
    
    def store_element_as_image(self, idx: int, output_path: Path, draw_bboxes: bool = False) -> None:
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

        # TODO: Add text there

        # TODO: Add `gt_encoded` here

def dummy_transform(img, aabbs):
    return img, aabbs