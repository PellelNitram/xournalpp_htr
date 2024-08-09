import gzip
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup as bs


@dataclass
class Page:
    """Class for keeping track of document page."""

    meta_data: dict
    background: dict
    layers: list


@dataclass
class Layer:
    """Class for keeping track of document page layer."""

    strokes: list


@dataclass
class Stroke:
    """Class for keeping track of strokes."""

    x: np.array
    y: np.array
    meta_data: dict


class Document(ABC):
    def __init__(self, path: Path):
        self.path = path
        self.pages = []
        self.DPI = -1
        self.load_data()

    @abstractmethod
    def load_data(self):
        """
        Loads data of document.

        Data comprises of stroke data on layers and pages as well as DPI.
        """
        pass

    def save_page_as_image(
        self,
        page_index: int,
        out_path: Path,
        black_white: bool = False,
        dpi: float = 72.0,
    ) -> Path:
        """
        Save document page as image.

        #TODO: I am using `matplotlib` here. Alternatively, OpenCV could do the trick as well.

        :param page_index: Index of page to save.
        :param output: Output path. Its file type determines output file type.
        :param black_white: Save image as black/white image if True.
        :param dpi: DPI of exported image.
        :returns: Output path.
        """
        p = self.pages[page_index]

        fig_width_inch = float(p.meta_data["width"]) / self.DPI
        fig_height_inch = float(p.meta_data["height"]) / self.DPI

        plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=self.DPI)
        for layer in p.layers:
            for stroke in layer.strokes:
                c = "black" if black_white else stroke.meta_data["color"]
                plt.plot(stroke.x / self.DPI, -stroke.y / self.DPI, c=c)
        plt.xlim(0, fig_width_inch)
        plt.ylim(-fig_height_inch, 0)
        plt.axis("off")
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
        plt.savefig(out_path, dpi=dpi)
        plt.close()

        return out_path


class XournalDocument(Document):
    def load_data(self):
        """Load Xournal document content."""

        with gzip.open(self.path, "r") as f:
            content = f.read().decode("utf-8")

        bs_content = bs(content, "lxml")

        for page in bs_content.find_all("page"):
            layers = []
            for layer in page.find_all("layer"):
                strokes = []
                for stroke in layer.find_all("stroke"):
                    x, y = np.fromstring(stroke.text, sep=" ").reshape(-1, 2).T
                    s = Stroke(x, y, stroke.attrs)
                    strokes.append(s)

                layers.append(Layer(strokes))

            background = page.find_all("background")
            assert len(background) == 1
            background = background[0].attrs

            p = Page(page.attrs, background, layers)

            self.pages.append(p)

        self.DPI = 72


class XournalppDocument(Document):
    def load_data(self):
        """Load Xournal document content."""

        with gzip.open(self.path, "r") as f:
            content = f.read().decode("utf-8")

        bs_content = bs(content, "lxml")

        for page in bs_content.find_all("page"):
            layers = []
            for layer in page.find_all("layer"):
                strokes = []
                for stroke in layer.find_all("stroke"):
                    x, y = np.fromstring(stroke.text, sep=" ").reshape(-1, 2).T
                    s = Stroke(x, y, stroke.attrs)
                    strokes.append(s)

                layers.append(Layer(strokes))

            background = page.find_all("background")
            assert len(background) == 1
            background = background[0].attrs

            p = Page(page.attrs, background, layers)

            self.pages.append(p)

        self.DPI = 72


def get_document(path: Path) -> Document:
    """
    Loads a document from a given file path based on the file extension.

    This function determines the appropriate document type to load by
    examining the file extension of the provided path. It supports files
    with the extensions `.xoj` and `.xopp`, returning a corresponding
    document object. If the file extension is not recognized, a
    `NotImplementedError` is raised.

    :param path: The file path to the document.
    :returns: An instance of the appropriate `Document` class. Either
              `XournalDocument` or `XournalppDocument`.
    :raises NotImplementedError: If the file extension is not supported.
                                 Supported file extensions are `.xoj` and
                                 `.xopp`.
    """

    file_ending = path.suffix

    if file_ending == ".xoj":
        document = XournalDocument(path)
    elif file_ending == ".xopp":
        document = XournalppDocument(path)
    else:
        raise NotImplementedError(
            f'File ending "{file_ending}" currently not readable.'
        )

    return document
