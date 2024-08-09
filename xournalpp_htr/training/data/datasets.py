"""
Module concerned with creating datasets for training custom
xournalpp_htr models.
"""

import os
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

from xournalpp_htr.training.io import load_IAM_OnDB_sample


class IAM_OnDB_Dataset(Dataset):
    """IAM-OnDB dataset implementation in PyTorch.

    These are the links to the dataset:
    - https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
    - https://doi.org/10.1109/ICDAR.2005.132

    This class encapsulates my own version of the IAM On-DB dataset in which I fixed a few small
    samples by fixing text formatting issues.

    This is the raw dataset which can be further processed using downstream transformations.
    """

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

                    df, text_line = load_IAM_OnDB_sample(sample_name, self.path)

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
