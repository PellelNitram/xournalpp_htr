"""
This script helps to annotate Xournal(++) files.
"""

# ====================
# Experiment w TKINTER
# ====================

# I use this to speed up the process

# TODO: Properly read about TKINTER interface design
# - https://tkinterpython.top/layout/

# TODO: Adhere to design document called `annotate_tool_UI_design.svg`. Potentially model it in PenPot?

# TODO: define storage schema to allow backward compatibility when improving the script later on

# -*- coding: utf-8 -*-
# Advanced zoom example. Like in Google Maps.
# It zooms only a tile, but not the whole image. So the zoomed tile occupies
# constant memory and not crams it with a huge resized image for the large zooms.
import dataclasses
import datetime
import json
import random
import sys
import tkinter as tk
import uuid
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from tkinter.filedialog import askopenfilename

import git
from PIL import Image, ImageTk

from xournalpp_htr.documents import XournalppDocument


class AutoScrollbar(ttk.Scrollbar):
    """A scrollbar that hides itself if it's not needed.
    Works only if you use the grid geometry manager"""

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError("Cannot use pack with this widget")

    def place(self, **kw):
        raise tk.TclError("Cannot use place with this widget")


class Zoom_Advanced(ttk.Frame):
    """Advanced zoom of the image"""

    def __init__(self, mainframe, path):
        """Initialize the main Frame"""
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title("Zoom with mouse wheel")
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient="vertical")
        hbar = AutoScrollbar(self.master, orient="horizontal")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="we")
        # Create canvas and put image on it
        self.canvas = tk.Canvas(
            self.master,
            highlightthickness=0,
            xscrollcommand=hbar.set,
            yscrollcommand=vbar.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nswe")
        self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.scroll_x)
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind("<Configure>", self.show_image)  # canvas is resized
        self.canvas.bind("<ButtonPress-1>", self.move_from)
        self.canvas.bind("<B1-Motion>", self.move_to)
        self.canvas.bind(
            "<MouseWheel>", self.wheel
        )  # with Windows and MacOS, but not Linux
        self.canvas.bind("<Button-5>", self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind("<Button-4>", self.wheel)  # only with Linux, wheel scroll up
        self.image = Image.open(path)  # open image
        self.width, self.height = self.image.size
        self.imscale = 1.0  # scale for the canvaas image
        self.delta = 1.3  # zoom magnitude
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(
            0, 0, self.width, self.height, width=0
        )
        # Plot some optional random rectangles for the test purposes
        minsize, maxsize, number = 5, 20, 10
        for _ in range(number):
            x0 = random.randint(0, self.width - maxsize)
            y0 = random.randint(0, self.height - maxsize)
            x1 = x0 + random.randint(minsize, maxsize)
            y1 = y0 + random.randint(minsize, maxsize)
            color = ("red", "orange", "yellow", "green", "blue")[random.randint(0, 4)]
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, activefill="black")
        self.show_image()

    def scroll_y(self, *args, **kwargs):
        """Scroll canvas vertically and redraw the image"""
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def scroll_x(self, *args, **kwargs):
        """Scroll canvas horizontally and redraw the image"""
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image

    def move_from(self, event):
        """Remember previous coordinates for scrolling with the mouse"""
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        """Drag (move) canvas to the new position"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image

    def wheel(self, event):
        """Zoom with mouse wheel"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            pass  # Ok! Inside the image
        else:
            return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30:
                return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale /= self.delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale:
                return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale *= self.delta
        self.canvas.scale("all", x, y, scale, scale)  # rescale all canvas objects
        self.show_image()

    def show_image(self, event=None):
        """Show image on the Canvas"""
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (
            self.canvas.canvasx(0),  # get visible area of the canvas
            self.canvas.canvasy(0),
            self.canvas.canvasx(self.canvas.winfo_width()),
            self.canvas.canvasy(self.canvas.winfo_height()),
        )
        bbox = [
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),  # get scroll region box
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3]),
        ]
        if (
            bbox[0] == bbox2[0] and bbox[2] == bbox2[2]
        ):  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if (
            bbox[1] == bbox2[1] and bbox[3] == bbox2[3]
        ):  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(
            bbox2[0] - bbox1[0], 0
        )  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if (
            int(x2 - x1) > 0 and int(y2 - y1) > 0
        ):  # show image if it in the visible area
            x = min(
                int(x2 / self.imscale), self.width
            )  # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            image = self.image.crop(
                (int(x1 / self.imscale), int(y1 / self.imscale), x, y)
            )
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(
                max(bbox2[0], bbox1[0]),
                max(bbox2[1], bbox1[1]),
                anchor="nw",
                image=imagetk,
            )
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = (
                imagetk  # keep an extra reference to prevent garbage-collection
            )


# path = "/home/martin/Downloads/apple_158989157.jpg"  # place path to your image here
# root = tk.Tk()
# app = Zoom_Advanced(root, path=path)
# root.mainloop()

# TODO: First, no zoom in!


# Useful structure for app: https://stackoverflow.com/questions/17125842/changing-the-text-on-a-label

root = tk.Tk()  # create root window
root.title("Annotate Tool")  # title of the GUI window
# root.maxsize(900, 600)  # specify the max size the window can expand to
root.geometry("1000x800")
root.config(bg="skyblue")  # specify background color

# TODO: Fix layout of GUI


DEBUG = len(sys.argv) > 1

if DEBUG:
    currently_loaded_document = Path(
        "/home/martin/Development/xournalpp_htr/tests/data/2024-07-26_minimal.xopp"
    )
else:
    currently_loaded_document = None


def load_document():
    global currently_loaded_document
    filename = askopenfilename()
    currently_loaded_document = filename
    status_file.configure(text=f"File loaded: {currently_loaded_document}")
    status_file.update()
    return filename


def draw_a_point(c: tk.Canvas, coord_x: float, coord_y: float, color: str) -> None:
    x1, y1 = (coord_x - 1), (coord_y - 1)
    x2, y2 = (coord_x + 1), (coord_y + 1)
    c.create_oval(x1, y1, x2, y2, fill=color)


I_PAGE = 0  # TODO: Make selectable

START_DRAWING_BBOX = False


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


BBOX_FIRST_POINT = None

LIST_OF_BBOXES = []


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


w = tk.Button(root, text="Load document", command=load_document)
w.place(x=50, y=50)

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


def listbox_select(event):
    index = listbox.curselection()[0]
    bbox = listbox.get(index, None)
    for bbox in LIST_OF_BBOXES:  # Reset colors
        canvas.itemconfig(bbox.rect_reference, outline=DEFAULT_BBOX_OUTLINE_COLOR)
    bbox: BBox = LIST_OF_BBOXES[index]
    canvas.itemconfig(bbox.rect_reference, outline=HIGHLIGHTED_BBOX_OUTLINE_COLOR)
    edit_text.delete(1.0, tk.END)
    edit_text.insert(tk.END, "" if bbox.text is None else bbox.text)


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


def update_bbox_text():
    index = listbox.curselection()[0]
    bbox = LIST_OF_BBOXES[index]
    bbox.text = edit_text.get("1.0", tk.END)


update_text = tk.Button(root, text="Update bbox text", command=update_bbox_text)
update_text.place(x=700, y=600)

annotator_ID = tk.Text(root, height=2, width=30, font=40)
annotator_ID.insert(tk.END, "(add annotator ID here)")
annotator_ID.place(x=800, y=650)

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
git_commit_hash_label = tk.Label(root, text=f"git commit: {sha}")
git_commit_hash_label.place(x=500, y=0)


# TODO: Next: drawing bounding box on canvas; and then keep track of them in listview
# - G"tkinter draw bounding box on canvas"
#   - https://stackoverflow.com/questions/29789554/tkinter-draw-rectangle-using-a-mouse

# todo: add button to start drawing bbox

# todo: add listview to show list of all bboxes

# todo: add details viewer to show a selected bbox, which consists of bbox and text

# todo: add button to export annotations; using a schema; export the minimal bbox (obviously)

# TODO: Add reference to drawn rectangle in order to change colour and to add annotated text.

# TODO: Add annotator's name. Do so by adding a text field.

# TODO: check why there's an exception (perceived somewhat randomly)

# # Create left and right frames
# left_frame = tk.Frame(root, width=200, height=400, bg="grey")
# left_frame.grid(row=0, column=0, padx=10, pady=5)

# right_frame = tk.Frame(root, width=650, height=400, bg="grey")
# right_frame.grid(row=0, column=1, padx=10, pady=5)

# # Create frames and labels in left_frame
# tk.Label(left_frame, text="Original Image").grid(row=0, column=0, padx=5, pady=5)

# # load image to be "edited"
# image = tk.PhotoImage(file="/home/martin/Downloads/small_VC.gif")
# original_image = image.subsample(3, 3)  # resize image using subsample
# tk.Label(left_frame, image=original_image).grid(row=1, column=0, padx=5, pady=5)

# # Display image in right_frame
# tk.Label(right_frame, image=image).grid(row=0, column=0, padx=5, pady=5)

# # Create tool bar frame
# tool_bar = tk.Frame(left_frame, width=180, height=185)
# tool_bar.grid(row=2, column=0, padx=5, pady=5)

# # Example labels that serve as placeholders for other widgets
# tk.Label(tool_bar, text="Tools", relief=tk.RAISED).grid(
#     row=0, column=0, padx=5, pady=3, ipadx=10
# )  # ipadx is padding inside the Label widget
# tk.Label(tool_bar, text="Filters", relief=tk.RAISED).grid(
#     row=0, column=1, padx=5, pady=3, ipadx=10
# )

# # Example labels that could be displayed under the "Tool" menu
# tk.Label(tool_bar, text="Select").grid(row=1, column=0, padx=5, pady=5)
# tk.Label(tool_bar, text="Crop").grid(row=2, column=0, padx=5, pady=5)
# tk.Label(tool_bar, text="Rotate & Flip").grid(row=3, column=0, padx=5, pady=5)
# tk.Label(tool_bar, text="Resize").grid(row=4, column=0, padx=5, pady=5)
# tk.Label(tool_bar, text="Exposure").grid(row=5, column=0, padx=5, pady=5)

root.mainloop()
