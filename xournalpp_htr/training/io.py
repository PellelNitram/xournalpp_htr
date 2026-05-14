"""Generic IO functionality."""

import json
from pathlib import Path


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
