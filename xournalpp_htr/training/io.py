"""Generic IO functionality."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
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


def store_list_of_bboxes(
    output_path: Path, list_of_bboxes: list, schema_version: str, meta_data: dict
):
    """TODO: Add test and docstring.

    TODO: Add schema to version data storage and loading properly. This is to make
    the anntation process future proof for upcoming `annotate.py` versions.
    """

    if schema_version == "v1_2024-10-13":
        storage = {
            "annotator_ID": meta_data["annotator_ID"],
            "writer_ID": meta_data["writer_ID"],
            "currently_loaded_document": meta_data["currently_loaded_document"],
            "page_index": meta_data["page_index"],
            "schema_version": "v1_2024-10-13",
            "bboxes": [],
        }

        for bbox in list_of_bboxes:
            value = {}

            # TODO: Use BBox.as_json_str; how does that work w/ `strokes` list?
            value["capture_date"] = str(bbox.capture_date)
            value["point_1_x"] = bbox.point_1_x
            value["point_1_y"] = bbox.point_1_y
            value["point_2_x"] = bbox.point_2_x
            value["point_2_y"] = bbox.point_2_y
            value["text"] = bbox.text
            value["uuid"] = bbox.uuid
            value["bbox_strokes"] = []
            for stroke in bbox.strokes:
                value["bbox_strokes"].append(
                    {
                        "meta_data": stroke.meta_data,
                        "x": stroke.x.tolist(),
                        "y": stroke.y.tolist(),
                    }
                )

            storage["bboxes"].append(value)

        with open(output_path, mode="w") as f:
            json.dump(storage, f)

    else:
        raise ValueError(f'"schema_version"={schema_version} not implemented')


def load_list_of_bboxes(input_path: Path) -> dict:
    """TODO: Add test and docstring and type annotations.

    See `store_list_of_bboxes`.
    """

    with open(input_path, mode="r") as f:
        data = json.load(f)

    schema_version = data["schema_version"]

    if schema_version == "v1_2024-10-13":
        result = {
            "annotator_ID": data["annotator_ID"],
            "writer_ID": data["writer_ID"],
            "currently_loaded_document": data["currently_loaded_document"],
            "page_index": data["page_index"],
            "schema_version": data["schema_version"],
            "bboxes": [],
        }

        for bbox in data["bboxes"]:
            bbox["capture_date"] = pd.to_datetime(bbox["capture_date"]).to_pydatetime()
            for bbox_stroke in bbox["bbox_strokes"]:
                bbox_stroke["x"] = np.array(bbox_stroke["x"])
                bbox_stroke["y"] = np.array(bbox_stroke["y"])
            result["bboxes"].append(bbox)

        return result

    else:
        raise ValueError(f'"schema_version"={schema_version} not implemented')
