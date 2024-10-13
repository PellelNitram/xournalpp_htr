"""This script helps to annotate Xournal++ files."""

import dataclasses
import datetime
import json
import sys
import tkinter as tk
import uuid
from dataclasses import dataclass
from pathlib import Path
from tkinter.filedialog import askopenfilename, asksaveasfilename

import git
import numpy as np

from xournalpp_htr.documents import Stroke, XournalppDocument

# =====
# TODOs
# =====

# TODO: Adhere to design document called `annotate_tool_UI_design.svg`. Potentially model it in PenPot?
# TODO: Useful structure for app: https://stackoverflow.com/questions/17125842/changing-the-text-on-a-label
# TODO: Improve GUI layout; Properly read about TKINTER interface design:
# - https://tkinterpython.top/layout/
# TODO: Everything in here is untested. I will probably leave it as is for now.

# ===========
# Helper code
# ===========


def load_document():
    global currently_loaded_document
    global I_PAGE
    I_PAGE = int(page_selector_text.get("1.0"))
    filename = askopenfilename()
    currently_loaded_document = filename
    status_file.configure(text=f"File loaded: {currently_loaded_document}")
    status_file.update()
    return filename


def draw_a_point(c: tk.Canvas, coord_x: float, coord_y: float, color: str) -> None:
    x1, y1 = (coord_x - 1), (coord_y - 1)
    x2, y2 = (coord_x + 1), (coord_y + 1)
    c.create_oval(x1, y1, x2, y2, fill=color)


@dataclass
class BBox:
    text: str
    point_1_x: float
    point_1_y: float
    point_2_x: float
    point_2_y: float
    capture_date: datetime.datetime
    uuid: str
    rect_reference: int | None
    strokes: list[Stroke] | None

    def __str__(self) -> str:
        return str(self.capture_date)

    def to_json_str(self) -> str:
        return json.dumps(
            dataclasses.asdict(self), indent=4, sort_keys=True, default=str
        )

    def from_json_str(self, json_str: str) -> None:
        print("from_json_str")
        pass

    @staticmethod
    def get_new_uuid() -> str:
        return str(uuid.uuid4())


def draw_document():
    canvas.delete("all")

    xpp_document = XournalppDocument(Path(currently_loaded_document))

    color = "black"  # python_green = "#476042"  # draw different colour for each stroke

    # Adjust canvas size
    coord_boundaries = xpp_document.get_min_max_coordinates_per_page()
    new_width = coord_boundaries[I_PAGE]["max_x"] * 1.1
    new_height = coord_boundaries[I_PAGE]["max_y"] * 1.1
    canvas.config(width=new_width, height=new_height)

    # Plot points
    for layer in xpp_document.pages[I_PAGE].layers:
        for stroke in layer.strokes:
            for coord_x, coord_y in zip(stroke.x, stroke.y):
                draw_a_point(canvas, coord_x, coord_y, color)

    if DRAW_STROKE_BOUNDING_BOX.get():
        x0 = coord_boundaries[I_PAGE]["min_x"]
        y0 = coord_boundaries[I_PAGE]["max_y"]
        x1 = coord_boundaries[I_PAGE]["max_x"]
        y1 = coord_boundaries[I_PAGE]["min_y"]
        canvas.create_rectangle(x0, y0, x1, y1, fill="", outline="red")
        canvas.create_text(x1, y1, text="data bounding box", anchor=tk.SE, fill="red")


def draw_bbox():
    global START_DRAWING_BBOX
    START_DRAWING_BBOX = True


def paint_bbox(event):
    global START_DRAWING_BBOX
    global BBOX_FIRST_POINT
    global LIST_OF_BBOXES
    if START_DRAWING_BBOX:
        BBOX_FIRST_POINT = event.x, event.y
        START_DRAWING_BBOX = False
    else:
        if BBOX_FIRST_POINT:
            # Get point
            second_point = event.x, event.y

            # Store bbox
            bbox = BBox(
                text=None,
                point_1_x=BBOX_FIRST_POINT[0],
                point_1_y=BBOX_FIRST_POINT[1],
                point_2_x=second_point[0],
                point_2_y=second_point[1],
                capture_date=datetime.datetime.now(),
                uuid=BBox.get_new_uuid(),
                rect_reference=None,
                strokes=None,
            )

            print(bbox)
            print(dataclasses.asdict(bbox))

            LIST_OF_BBOXES.append(bbox)

            # Draw
            rect = canvas.create_rectangle(
                bbox.point_1_x,
                bbox.point_1_y,
                bbox.point_2_x,
                bbox.point_2_y,
                fill="",
                outline=DEFAULT_BBOX_OUTLINE_COLOR,
            )
            bbox.rect_reference = rect
            print(
                rect, type(rect)
            )  # Use it like shown here: https://stackoverflow.com/a/35935638 & https://stackoverflow.com/a/13212501

            # Add to listview
            listbox.insert(tk.END, bbox)

            # Book keeping
            BBOX_FIRST_POINT = None


def listbox_select(event):
    index = listbox.curselection()[0]
    bbox = listbox.get(index, None)
    for bbox in LIST_OF_BBOXES:  # Reset colors
        canvas.itemconfig(bbox.rect_reference, outline=DEFAULT_BBOX_OUTLINE_COLOR)
    bbox: BBox = LIST_OF_BBOXES[index]
    canvas.itemconfig(bbox.rect_reference, outline=HIGHLIGHTED_BBOX_OUTLINE_COLOR)
    edit_text.delete(1.0, tk.END)
    edit_text.insert(tk.END, "" if bbox.text is None else bbox.text)


def update_bbox_text():
    index = listbox.curselection()[0]
    bbox = LIST_OF_BBOXES[index]
    bbox.text = edit_text.get("1.0", tk.END).strip()


def export():
    if DEBUG:
        output_path = Path(
            "/home/martin/Development/xournalpp_htr/tests/data/2024-10-12_annotate_test_output.json"
        )
    else:
        output_path = Path(
            asksaveasfilename(
                initialfile="Untitled.json",
                defaultextension=".json",
                filetypes=[("All Files", "*.*"), ("JSON Documents", "*.json")],
            )
        )

    xpp_document = XournalppDocument(Path(currently_loaded_document))

    # Get all strokes on that page in a list
    all_strokes: list[Stroke] = []
    for layer in xpp_document.pages[I_PAGE].layers:
        for stroke in layer.strokes:
            all_strokes.append(stroke)

    stroke_already_used = np.zeros(len(all_strokes), dtype=bool)

    # Determine strokes
    for bbox in LIST_OF_BBOXES:
        min_x = min(bbox.point_1_x, bbox.point_2_x)
        max_x = max(bbox.point_1_x, bbox.point_2_x)
        min_y = min(bbox.point_1_y, bbox.point_2_y)
        max_y = max(bbox.point_1_y, bbox.point_2_y)
        bbox_strokes = []
        for i_stroke, stroke in enumerate(all_strokes):
            # Skip strokes that are already part of a bbox
            if stroke_already_used[i_stroke]:
                continue
            condition_x_min = np.all(min_x <= stroke.x)
            condition_x_max = np.all(stroke.x <= max_x)
            condition_y_min = np.all(min_y <= stroke.y)
            condition_y_max = np.all(stroke.y <= max_y)
            if (
                condition_x_min
                and condition_x_max
                and condition_y_min
                and condition_y_max
            ):
                bbox_strokes.append(stroke)
                stroke_already_used[i_stroke] = True
        # assert bbox.strokes is None
        bbox.strokes = bbox_strokes

        # TODO: Store bbox'es as JSON; I do so by storing all the
        # relevant detail in a dict first. The method that turns
        # it in to a JSON might as well be part of `Bbox`!

        # TODO: Also loading the JSON into a dict will be a part of
        # the `Bbox` class for the sake of simplicity.

        # TODO: Add a storage schema
        # TODO: define storage schema to allow backward compatibility when improving the script later on

    storage = {"bboxes": []}

    for bbox in LIST_OF_BBOXES:
        print(bbox)
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
        storage["annotator_ID"] = annotator_ID.get("1.0", tk.END).strip()
        storage["writer_ID"] = writer_ID.get("1.0", tk.END).strip()
        storage["currently_loaded_document"] = str(currently_loaded_document)
        storage["page_index"] = I_PAGE

    with open(output_path, mode="w") as f:
        json.dump(storage, f)


# =========
# Main code
# =========


root = tk.Tk()  # create root window
root.title("Annotate Tool")  # title of the GUI window
root.geometry("1000x800")
root.config(bg="skyblue")  # specify background color


DEBUG = len(sys.argv) > 1

if DEBUG:
    currently_loaded_document = Path(
        "/home/martin/Development/xournalpp_htr/tests/data/2024-07-26_minimal.xopp"
    )
else:
    currently_loaded_document = None


I_PAGE = 0

START_DRAWING_BBOX = False


BBOX_FIRST_POINT = None

LIST_OF_BBOXES: list[BBox] = []


w = tk.Button(root, text="Load document", command=load_document)
w.place(x=50, y=50)

page_selector_label = tk.Label(root, text="Select page:")
page_selector_label.place(x=200, y=50)

page_selector_text = tk.Text(
    root,
    height=1,
    width=4,
    font=40,
)
page_selector_text.place(x=280, y=50)
page_selector_text.insert("1.0", "0")

DRAW_STROKE_BOUNDING_BOX = tk.BooleanVar()
b = tk.Checkbutton(
    root, text="Enable DRAW_STROKE_BOUNDING_BOX?", variable=DRAW_STROKE_BOUNDING_BOX
)
b.place(x=50, y=120)

button_draw = tk.Button(root, text="Draw document", command=draw_document)
button_draw.place(x=50, y=90)

status_file = tk.Label(root, text=f"File loaded: {currently_loaded_document}")
status_file.place(x=0, y=20)

status_bar = tk.Label(root, text="status bar")
status_bar.place(x=0, y=0)

canvas = tk.Canvas(root, width=500, height=500)
canvas.place(x=50, y=150)
canvas.bind("<Button-1>", paint_bbox)

button_draw_bbox = tk.Button(root, text="Draw bbox", command=draw_bbox)
button_draw_bbox.place(x=200, y=90)

DEFAULT_BBOX_OUTLINE_COLOR = "orange"
HIGHLIGHTED_BBOX_OUTLINE_COLOR = "red"


# create listbox object
listbox = tk.Listbox(
    root,
    height=10,
    width=25,
    bg="grey",
    activestyle="dotbox",
    font="Helvetica",
    fg="yellow",
)
listbox.place(x=700, y=150)
# See here for what I want to do: https://tk-tutorial.readthedocs.io/en/latest/listbox/listbox.html#edit-a-listbox-item
listbox.bind("<<ListboxSelect>>", listbox_select)
# Another good resource: https://www.geeksforgeeks.org/python-tkinter-listbox-widget/


edit_text = tk.Text(root, height=2, width=30, font=40)
edit_text.place(x=700, y=500)


update_text = tk.Button(root, text="Update bbox text", command=update_bbox_text)
update_text.place(x=700, y=600)

annotator_ID = tk.Text(root, height=2, width=30, font=40)
annotator_ID.insert(tk.END, "(add annotator ID here)")
annotator_ID.place(x=800, y=650)

writer_ID = tk.Text(root, height=2, width=30, font=40)
writer_ID.insert(tk.END, "(add writer ID here)")
writer_ID.place(x=800, y=700)

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
git_commit_hash_label = tk.Label(root, text=f"git commit: {sha}")
git_commit_hash_label.place(x=500, y=0)


export_annotations = tk.Button(root, text="Export annotations", command=export)
export_annotations.place(x=50, y=750)

root.mainloop()
