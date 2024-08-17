"""
Module concerned with creating datasets for training custom
xournalpp_htr models.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

DatasetIndex = int


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
    :type stroke_width: float
    :param page_index: The index of the page where this position is located.
    :type page_index: int
    :param center_x: The x-coordinate of the center of the position.
    :type center_x: float
    :param center_y: The y-coordinate of the center of the position.
    :type center_y: float
    :param height: The height of the position. The width is derived based on this (using sample) to maintain aspect ratio.
    :type height: float
    """

    stroke_width: float
    page_index: int
    center_x: float
    center_y: float
    height: float  # width is derived automatically by keeping aspect ratio constant


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
        positions: dict[DatasetIndex, PageDatasetFromOnlinePosition],
        page_size: list[float, float],  # TODO: Think about unit! mm, inch, dots?
    ) -> None:
        """Initialise a `PageDataset`.

        TODO.
        """
        self.dataset = dataset
        self.positions = positions
        self.page_size = page_size

    def compute(self) -> list:
        pass
        # Steps to perform:
        # 1. Loop over `self.positions` to obtain index and location.
        # 2. Get sample from dataset.
        # 3. Transform sample location to reflect position on page.
        # 4. Store all that in a list and return it

    def render_pages(self) -> None:
        pass
        # Steps to perform:
        # TODO

    # TODO: When placing the positions, the dataset should spit out a warning,
    #       or crash, if bounding boxes overlap b/c that'd never happen for a
    #       normal document
