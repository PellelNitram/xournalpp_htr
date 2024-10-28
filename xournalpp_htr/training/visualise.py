"""Visualisation codes for training purposes."""

import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
import pandas as pd


def plot_clustered_document(
    a_ground_truth: mpl.axis.Axis,
    a_predicted: mpl.axis.Axis,
    clustering,
    annotated_bboxes,
    DPI,
    df_train: pd.DataFrame,
    a_predicted_title: str,
) -> None:
    """Plots clustered document into axes.

    TODO: explain.

    TODO: docstring and type annotations.
    """

    # ===================
    # Ground truth figure
    # ===================

    # I replicated this from below - TODO: Follow DIY and consolidate

    a_ground_truth.set_aspect("equal")
    a_ground_truth.set_xlabel("x")
    a_ground_truth.set_ylabel("y")

    for i_bbox in range(len(annotated_bboxes["bboxes"])):
        bbox = annotated_bboxes["bboxes"][i_bbox]

        # Draw bbox
        xy = (
            min([bbox["point_1_x"], bbox["point_2_x"]]) / DPI,
            min([-bbox["point_1_y"], -bbox["point_2_y"]])
            / DPI,  # TODO: This messing around w/ y coord sign is annoying
        )
        dx = np.abs(bbox["point_1_x"] - bbox["point_2_x"]) / DPI
        dy = np.abs(bbox["point_1_y"] - bbox["point_2_y"]) / DPI
        a_ground_truth.add_patch(
            patches.Rectangle(xy, dx, dy, linewidth=1, edgecolor="r", facecolor="none")
        )

        # Draw label
        a_ground_truth.text(x=xy[0], y=xy[1] + dy, s=bbox["text"], c="red")

        for bbox_stroke in bbox["bbox_strokes"]:
            x = bbox_stroke["x"] / DPI
            y = bbox_stroke["y"] / DPI
            a_ground_truth.scatter(x, -y, c="black", s=1)

    # ================
    # Predicted figure
    # ================

    a_predicted.set_aspect("equal")
    a_predicted.set_xlabel("x")
    a_predicted.set_ylabel("y")
    a_predicted.set_title(a_predicted_title)

    for i_cluster in np.unique(clustering.labels_):
        stroke_indices = np.where(clustering.labels_ == i_cluster)[0]

        print(i_cluster, stroke_indices)

        x_coords = []
        y_coords = []
        x_coords_mean = []
        y_coords_mean = []
        for stroke_index in stroke_indices:
            stroke_row = df_train.iloc[stroke_index]

            x_coords += stroke_row["x"].tolist()
            y_coords += stroke_row["y"].tolist()
            x_coords_mean.append(stroke_row["x_mean"])
            y_coords_mean.append(stroke_row["y_mean"])

        a_predicted.scatter(x_coords, y_coords, s=1)
        a_predicted.scatter(x_coords_mean, y_coords_mean, c="red", s=1)
