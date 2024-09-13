"""Generic IO functionality."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


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


def store_alphabet(outfile: Path, alphabet: list[str]) -> None:
    """Stores the alphabet as JSON.

    :param outfile: The path to store the alphabet under.
    :param alphabet: The alphabet.
    """
    with open(outfile, "w") as f:
        json.dump({"alphabet": alphabet}, f, indent=4)


def load_alphabet(infile: Path) -> list[str]:
    """Load alphabet from JSON.

    :param infile: The path to load the alphabet from.
    :returns: The alphabet as list of strings.
    """
    with open(infile, "r") as f:
        json_data = json.load(f)
    return json_data["alphabet"]
