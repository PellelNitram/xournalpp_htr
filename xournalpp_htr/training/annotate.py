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
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.filedialog import askopenfilename

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

DRAW_STROKE_BOUNDING_BOX = True


def draw_document():
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

    if DRAW_STROKE_BOUNDING_BOX:
        x0 = coord_boundaries[I_PAGE]["min_x"]
        y0 = coord_boundaries[I_PAGE]["max_y"]
        x1 = coord_boundaries[I_PAGE]["max_x"]
        y1 = coord_boundaries[I_PAGE]["min_y"]
        canvas.create_rectangle(x0, y0, x1, y1, fill="", outline="red")
        canvas.create_text(x1, y1, text="data bounding box", anchor=tk.SE, fill="red")


w = tk.Button(root, text="Load document", command=load_document)
w.place(x=50, y=50)

button_draw = tk.Button(root, text="Draw document", command=draw_document)
button_draw.place(x=50, y=90)

status_file = tk.Label(root, text=f"File loaded: {currently_loaded_document}")
status_file.place(x=0, y=20)

status_bar = tk.Label(root, text="status bar")
status_bar.place(x=0, y=0)

canvas = tk.Canvas(root, width=500, height=500)
canvas.place(x=50, y=150)

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
