"""
Module concerned with creating datasets for training custom
xournalpp_htr models.
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

DatasetIndex = int
MillimeterDimension = float
DotDimension = int


class IAM_OnDB_Dataset(Dataset):
    """IAM-OnDB dataset implementation in PyTorch.

    These are the links to the dataset:
    - https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
    - https://doi.org/10.1109/ICDAR.2005.132

    This class encapsulates my own version of the IAM On-DB dataset in which I fixed a few small
    samples by fixing text formatting issues.

    This is the raw dataset which can be further processed using downstream transformations.
    """

    @staticmethod
    def load_df_iam_ondb(path: Path) -> pd.DataFrame:
        """
        Load IAM OnDB strokes file as pd.DataFrame.

        Does not read any meta data.

        :param path: The path to the XML strokes file.
        :returns: pd.DataFrame with strokes stored in columns "x", "y", "t" and "stroke_nr".
        """
        tree = ET.parse(path)
        root = tree.getroot()

        for element in root:
            if element.tag == "StrokeSet":
                stroke_set = element
                break

        data = {"x": [], "y": [], "t": [], "stroke_nr": []}

        stroke_nr = 0
        for stroke in stroke_set:
            for point in stroke:
                data["x"].append(float(point.attrib["x"]))
                data["y"].append(float(point.attrib["y"]))
                data["t"].append(float(point.attrib["time"]))
                data["stroke_nr"].append(stroke_nr)
            stroke_nr += 1

        df = pd.DataFrame.from_dict(data)

        assert df["stroke_nr"].max() + 1 == len(stroke_set)

        return df

    @staticmethod
    def load_IAM_OnDB_text_line(path: Path, line_nr: int) -> str:
        """
        Load text line of IAM OnDB sample.

        :param path: Path to lines file.
        :param line_nr: Number of line to extract. This is a 0-indexed value.
        :returns: The text line.
        """

        with open(path, "r") as f:
            all_lines = [xx.strip() for xx in f.readlines()]

        for ii, line in enumerate(all_lines):
            if line == "CSR:":
                i_start = ii + 1
                break

        all_lines = all_lines[i_start:]
        all_lines = [xx for xx in all_lines if len(xx) > 0]

        return all_lines[line_nr]

    @staticmethod
    def load_IAM_OnDB_sample(sample, base_path):
        """
        Load IAM On-DB data sample.

        With sample consisting of time series and text line as ground truth.

        :param sample: Sample code according to IAM On-DB encoding.
        :param base_path: Base path of IAM On-DB.
        :returns: (df, text_line) with df as time series.
        """

        SPLITTER = "-"

        code1, code2, code3 = sample.split(SPLITTER)
        code2_no_letters = "".join(
            [letter for letter in code2 if letter in "0123456789"]
        )

        strokes_file = Path(
            base_path
            / f"lineStrokes-all/lineStrokes/{code1}/{code1}{SPLITTER}{code2_no_letters}/{code1}{SPLITTER}{code2}{SPLITTER}{code3}.xml"
        )
        text_line_file = Path(
            base_path
            / f"ascii-all/ascii/{code1}/{code1}{SPLITTER}{code2_no_letters}/{code1}{SPLITTER}{code2}.txt"
        )

        df = IAM_OnDB_Dataset.load_df_iam_ondb(strokes_file)
        df["y"] *= -1  # Correct text direction to natural direction facing upwards

        text_line = IAM_OnDB_Dataset.load_IAM_OnDB_text_line(
            text_line_file, int(code3) - 1
        )

        return df, text_line

    LENGTH = 12187  # Determined empirically

    SAMPLES_NOT_TO_STORE = [
        "z01-000z-01",  # There exists no text for that sample
        "z01-000z-02",  # There exists no text for that sample
        "z01-000z-03",  # There exists no text for that sample
        "z01-000z-04",  # There exists no text for that sample
        "z01-000z-05",  # There exists no text for that sample
        "z01-000z-06",  # There exists no text for that sample
        "z01-000z-07",  # There exists no text for that sample
        "z01-000z-08",  # There exists no text for that sample
    ]

    # These are the samples that fail when transformed with Carbune2020 transformation.
    # I determined the samples here empirically.
    SAMPLES_TO_SKIP_BC_CARBUNE2020_FAILS = [
        "c02-082-06",
        "c02-082-02",
        "p04-468z-01",
        "e04-026-01",
        "e04-083-06",
        "e04-083-03",
        "e04-083-02",
        "a01-007w-01",
        "a01-087-02",
        "a01-020x-03",
        "a01-020x-04",
        "a01-053-03",
        "a01-053-04",
        "a01-053x-01",
        "a01-053x-03",
        "p09-110z-06",
        "d04-125-05",
        "h01-030-03",
        "a02-037-04",
        "a02-102-02",
        "a02-017-05",
        "a02-120-03",
        "k08-835z-01",
        "c08-465z-07",
        "c04-134-01",
        "c04-061-01",
        "g01-004-05",
        "h02-037-02",
        "h02-037-03",
        "h02-024-04",
        "j08-408z-05",
        "b04-134-04",
        "g06-000n-02",
        "g06-000k-03",
        "g06-000i-06",
        "g06-000k-01",
        "g06-000i-09",
        "g06-000f-04",
        "h07-260z-05",
        "h07-013-02",
        "m05-480z-01",
        "m05-538z-05",
        "a04-047-03",
        "b05-032-02",
        "b05-032-03",
        "g07-065-02",
        "g03-026-03",
        "d01-024-02",
        "f07-028b-02",
        "f07-028b-05",
        "f03-222z-03",
        "f03-174-04",
        "j04-015-02",
        "g04-055-02",
        "n01-051z-05",
        "l06-644z-02",
        "g10-343z-07",
        "a06-070-03",
        "a06-114-06",
        "a06-014-04",
        "a06-064-04",
        "f04-083-01",
        "j01-049-03",
        "j01-049-02",
        "j01-007z-07",
        "j01-063-01",
        "n10-293z-02",
    ]

    def __init__(
        self,
        path: Path,
        transform=None,
        limit: int = -1,
        skip_carbune2020_fails: bool = False,
    ) -> None:
        """Initialise an `IAM_OnDB_Dataset`.

        The data of the dataset needs to be stored on disk as follows to be readable by this present class:
        - `path`/lineStrokes-all/lineStrokes/
        - `path`/ascii-all/ascii/

        :param path: Path to dataset.
        :param limit: Limit number of loaded samples to this value if positive.
        :param skip_carbune2020_fails: Skip all sample that are known to fail when the `Carbune2020` transform is applied.
        """
        self.path = path
        self.transform = transform
        self.limit = limit
        self.skip_carbune2020_fails = skip_carbune2020_fails
        self.data = self.load_data()

    def load_data(self) -> List:
        """
        Returns IAM-OnDB data.

        In `__init__`, it is saved as `self.data`.

        Loading is performed by parsing the XML files and reading the text files.
        """

        result = []

        ctr = 0  # Starts at 1

        ended = False

        for _, _, files in tqdm(
            os.walk(self.path / "lineStrokes-all"),
            desc="Load data for IAM_OnDB_Dataset",
        ):
            if ended:
                break

            for f in files:
                if f.endswith(".xml"):
                    sample_name = f.replace(".xml", "")

                    if self.limit >= 0 and ctr >= self.limit:
                        ended = True
                        break

                    if sample_name in self.SAMPLES_NOT_TO_STORE:
                        continue

                    if (
                        self.skip_carbune2020_fails
                        and sample_name in self.SAMPLES_TO_SKIP_BC_CARBUNE2020_FAILS
                    ):
                        continue

                    df, text_line = IAM_OnDB_Dataset.load_IAM_OnDB_sample(
                        sample_name, self.path
                    )

                    result.append(
                        {
                            "x": df["x"].to_numpy(),
                            "y": df["y"].to_numpy(),
                            "t": df["t"].to_numpy(),
                            "stroke_nr": list(df["stroke_nr"]),
                            "label": text_line,
                            "sample_name": sample_name,
                        }
                    )

                    ctr += 1

        result.sort(key=lambda sample: sample["sample_name"])

        return result

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def plot_sample_to_image_file(self, sample_index: int, file_path: Path) -> None:
        """Plot sample data to image file.

        Helpful for debugging. It uses the `__getitem__` function and thereby applies transforms.

        :param sample_index: Index of sample to plot.
        :param file_path: Path to store image file as. Needs to come with suffix (this is not checked).
        """

        sample = self[sample_index]

        plt.figure()
        plt.scatter(
            sample["x"],
            sample["y"],
            c=sample["stroke_nr"],
            s=1,
            cmap=matplotlib.colormaps.get_cmap("Set1"),
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{sample['sample_name']}: {sample['label']}")
        plt.gca().set_aspect("equal")
        plt.savefig(file_path)
        plt.close()


@dataclass
class PageDatasetFromOnlinePosition:
    """Represents a position of a handwritten text placed on a page.

    This class stores information about a specific position within a page, including its stroke width,
    page index, center coordinates and height. The width is automatically derived to maintain a constant aspect ratio.

    :param stroke_width: The width of the stroke used at this position.
    :type stroke_width: MillimeterDimension
    :param page_index: The index of the page where this position is located.
    :type page_index: int
    :param center_x: The x-coordinate of the center of the position.
    :type center_x: MillimeterDimension
    :param center_y: The y-coordinate of the center of the position.
    :type center_y: MillimeterDimension
    :param height: The height of the position. The width is derived based on this (using sample) to maintain aspect ratio.
    :type height: MillimeterDimension
    :param dataset_index: The index in the dataset that this assigned this position.
    :type datset_index: DatasetIndex
    """

    stroke_width: MillimeterDimension
    page_index: int
    center_x: MillimeterDimension
    center_y: MillimeterDimension
    height: MillimeterDimension  # width is derived automatically by keeping aspect ratio constant
    dataset_index: DatasetIndex


class PageDatasetFromOnline(Dataset):
    """Dataset to assemble a page dataset using samples from an existing dataset.

    It places existing samples on a page while keeping track of the positions of
    the bounding boxes.

    TODO.
    """

    # TODO: Think about how to store both images and

    # TODO: Will keep track of it in online space and then can render to offline space

    # TODO: Check if a placed sample leaves the page and also if sample overlap with existing ones.

    def __init__(
        self,
        dataset: Dataset,  # TODO: An online dataset; can come w/ a transform obviously if desired
        positions: list[PageDatasetFromOnlinePosition],
        page_size: list[MillimeterDimension, MillimeterDimension],
        cache_dir: Path,
        dpi: float,
    ) -> None:
        """Initialise a `PageDataset`.

        TODO.
        """
        self.dataset = dataset
        self.positions = positions
        self.page_size = page_size
        self.cache_dir = cache_dir
        self.dpi = dpi
        self.data = self.compute()
        PageDatasetFromOnline.check_if_bounding_boxes_overlap(self.data)

    def compute(self) -> defaultdict[list]:
        """TODO.

        These are the steps performed:
        1. Loop over `self.positions` to obtain index and location.
        2. Get sample from dataset.
        3. Transform sample location to reflect position on page.
        4. Store all that in a list and return it, including the label.

        TODO: Explain returned list.
        """
        result = defaultdict(list)
        for position in self.positions:
            sample = self.dataset[position.dataset_index]
            x = sample["x"]
            x = x - x.min()
            y = sample["y"]
            y = y - y.min()
            label = sample["label"]
            scale_factor = position.height / y.max()
            x *= scale_factor
            y *= scale_factor
            x = x - x.max() / 2 + position.center_x
            y = y - y.max() / 2 + position.center_y
            stroke_nrs = np.sort(np.unique(sample["stroke_nr"]))
            strokes = {}
            for stroke_nr in stroke_nrs:
                mask = sample["stroke_nr"] == stroke_nr
                strokes[stroke_nr] = {
                    "x": x[mask].copy(),
                    "y": y[mask].copy(),
                }
            result[position.page_index].append(
                {
                    "strokes": strokes,
                    "label": label,
                    "stroke_width": position.stroke_width,
                    "center_x": position.center_x,
                    "center_y": position.center_y,
                }
            )
        return result

    def __len__(self) -> int:
        return len(self.data)

    def render_page_and_mask(
        self, page_index: int, output_path_page: Path, output_path_mask: Path
    ) -> None:
        """TODO.

        Steps that are performed: TODO.

        TODO: Determine page sizes etc & adjust rendering
        """

        inch_per_mm = 1.0 / 25.4
        dots_per_mm = self.dpi * inch_per_mm

        image_size = (
            round(self.page_size[1] * dots_per_mm),
            round(self.page_size[0] * dots_per_mm),
        )
        im_page = Image.new(
            "RGB",
            image_size,
            "white",
        )
        im_mask = Image.new(
            "RGB",
            image_size,
            "white",
        )
        draw_page = ImageDraw.Draw(im_page)
        draw_mask = ImageDraw.Draw(im_mask)
        for data in self.data[page_index]:
            x0 = np.inf
            y0 = np.inf
            x1 = -np.inf
            y1 = -np.inf
            for stroke_nr in data["strokes"]:
                x = +data["strokes"][stroke_nr]["x"] * dots_per_mm
                y = (
                    -1 * data["strokes"][stroke_nr]["y"] + 2 * data["center_y"]
                ) * dots_per_mm
                # `y` needs modification because PIL y direction points downwards,
                # so that the data appears mirrored. The `y` data modification is
                # a transform that is based on the idea to flip on y mean of
                # bounding box axis. TODO: Add blog article on that here where I explain
                # how to construct such a data transform.
                draw_page.line(
                    list(zip(x, y)),
                    fill="black",
                    width=round(data["stroke_width"] * dots_per_mm),
                )
                x0, y0, x1, y1 = PageDatasetFromOnline.compute_segmentation_masks(
                    x, y, x0, y0, x1, y1
                )
            draw_mask.rectangle(xy=[(x0, y0), (x1, y1)], fill="black")
        im_page.save(output_path_page)
        im_mask.save(output_path_mask)

        # TODO: Should I maybe go back to dots as unit? just to have more
        # control w/o `round()` function. I prefer control at the level of
        # my training data as to be able to reproduce it easily.

    @staticmethod
    def compute_segmentation_masks(
        x, y, x0, y0, x1, y1
    ) -> tuple[float, float, float, float]:
        """TODO."""
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        if x_min < x0:
            x0 = x_min
        if x_max > x1:
            x1 = x_max
        if y_min < y0:
            y0 = y_min
        if y_max > y1:
            y1 = y_max
        return x0, y0, x1, y1

    @staticmethod
    def get_file_name(idx: int, file_type: str) -> str:
        return f"{file_type}_{idx:06}.png"

    def __getitem__(self, idx: int) -> dict:
        """TODO.

        TODO: Idea behind logic:
        - once accessed, the page is rendered and saved in output folder (which is specified to constructor)
        - then, it is returned
        - then, when accessed again, it is loaded from page instead of recomputed

        TODO.
        """
        filename_page = self.cache_dir / PageDatasetFromOnline.get_file_name(
            idx, "page"
        )
        filename_mask = self.cache_dir / PageDatasetFromOnline.get_file_name(
            idx, "mask"
        )

        if not filename_page.exists() or not filename_mask.exists():
            self.render_page_and_mask(idx, filename_page, filename_mask)

        image = read_image(filename_page)
        segmentation_mask = read_image(filename_mask)  # TODO: Test it!

        sample = {
            "image": image,
            "segmentation_mask": segmentation_mask,
        }

        return sample

    @staticmethod
    def check_if_bounding_boxes_overlap(data):
        """TODO

        raises an error if they overlap b/c that is not allowed as it's not possible in a document
        that I consider here.

        TODO: Do a pairwise check, also stating that this leads to O(N^2) unfortunately.

        When placing the positions, the dataset should spit out a warning,
        or crash, if bounding boxes overlap b/c that'd never happen for a
        normal document -> this is what the method here checks for and it
        raises an exception if they overlap!
        """

        # TODO: Write test for function!

        def does_bboxes_overlap(bbox_1, bbox_2):
            """TODO.

            Sources:
            - https://stackoverflow.com/a/40795835 and
            - https://code.tutsplus.com/collision-detection-using-the-separating-axis-theorem--gamedev-169t

            note that I do allow the bounding boxes to be directly adjacent to each others
            """
            separated_by_x = (
                bbox_1["top_right_x"] <= bbox_2["bottom_left_x"]
                or bbox_2["top_right_x"] <= bbox_1["bottom_left_x"]
            )
            separated_by_y = (
                bbox_1["top_right_y"] <= bbox_2["bottom_left_y"]
                or bbox_2["top_right_y"] <= bbox_1["bottom_left_y"]
            )
            return not (separated_by_x or separated_by_y)

        # First, get bounding boxes for every page and position
        bounding_boxes = []
        for page_index in data:
            positions = data[page_index]
            for i_position, position in enumerate(positions):
                all_strokes_x = []
                all_strokes_y = []

                for stroke in position["strokes"]:
                    data_x = position["strokes"][stroke]["x"]
                    data_y = position["strokes"][stroke]["y"]
                    all_strokes_x.extend(data_x)
                    all_strokes_y.extend(data_y)
                bbox = {
                    "bottom_left_x": min(all_strokes_x),
                    "bottom_left_y": min(all_strokes_y),
                    "top_right_x": max(all_strokes_x),
                    "top_right_y": max(all_strokes_y),
                }
                bounding_boxes.append(
                    {"page_index": page_index, "i_position": i_position, "bbox": bbox}
                )

        # Second, check if the boxes intersect in a pairwise manner
        for i in range(len(bounding_boxes)):
            for j in range(i + 1, len(bounding_boxes)):
                data_1 = bounding_boxes[i]
                data_2 = bounding_boxes[j]
                bbox_1 = data_1["bbox"]
                bbox_2 = data_2["bbox"]
                if does_bboxes_overlap(bbox_1, bbox_2):
                    raise ValueError(
                        f"bounding boxes may not overlap but do: {data_1}, {data_2}"
                    )
