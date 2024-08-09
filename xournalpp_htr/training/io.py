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
