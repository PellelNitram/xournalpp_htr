# Architecture

This page describes the architecture of Xournal++ HTR from the perspective of its three user-facing entry points: the **CLI**, the **Xournal++ plugin**, and the **web demo**.

## Overview

All three entry points feed into the same core HTR pipeline, which converts a Xournal++ document (`.xoj`/`.xopp`) into a searchable PDF:

```mermaid
graph LR
    A["Xournal++ Plugin<br/>(Lua)"] -->|subprocess| CLI
    CLI["CLI<br/>(run_htr.py)"] --> P["HTR Pipeline<br/>(shortcuts.py)"]
    D["Web Demo<br/>(Gradio)"] --> P
    P --> PDF["Searchable PDF"]
```

## Entry Points

### CLI (`xournalpp_htr/run_htr.py`)

The command-line interface. Parses arguments via `utils.parse_arguments()` and delegates to `shortcuts.export_xournalpp_to_pdf_with_htr()`.

```bash
python xournalpp_htr/run_htr.py \
    -if input.xopp \
    -of output.pdf \
    [-m MODEL_NAME] \
    [-pid PREDICTION_IMAGE_DIR] \
    [-sp]
```

### Xournal++ Plugin (`plugin/main.lua`)

A Lua plugin that integrates into Xournal++ as a menu item (`Tools > Xournal++ HTR`, shortcut `Ctrl+F1`). It prompts the user for a save location and then calls `run_htr.py` via `os.execute`. Configuration (Python path, script path, model) is stored in `plugin/config.lua`.

### Web Demo (`scripts/demo.py`)

A Gradio app deployed as a HuggingFace Space (via `Dockerfile`). It provides a browser-based UI where users upload `.xoj`/`.xopp` files and step through the pipeline interactively. Unlike the CLI and plugin, the demo calls the pipeline functions directly (not via `shortcuts.py`) and displays the first page as a preview. It optionally logs interactions to Supabase for analytics and data donation.

## Core Pipeline

The pipeline in `shortcuts.py` runs three sequential steps:

```mermaid
graph TD
    INPUT[".xoj / .xopp file"] --> PIPELINE["export_xournalpp_to_pdf_with_htr()"]

    subgraph "Step 1: Export"
        PIPELINE --> S1["export_to_pdf_with_xournalpp()"]
        S1 -->|"xournalpp CLI"| TMP["Temporary PDF<br/>(no text layer)"]
    end

    subgraph "Step 2: Recognise"
        PIPELINE --> DOC["get_document()"]
        DOC -->|"Document object"| CP["compute_predictions()"]
        CP --> RENDER["save_page_as_image()<br/>(matplotlib, 150 DPI)"]
        RENDER --> HTR["read_page()<br/>(htr_pipeline)"]
        HTR --> PRED["Predictions dict"]
    end

    subgraph "Step 3: Embed"
        PIPELINE --> WRITE["write_predictions_to_PDF()"]
        TMP --> WRITE
        PRED --> WRITE
        WRITE -->|"PyMuPDF"| OUT["Output PDF<br/>(with text layer)"]
    end
```

### Step 1: Export to PDF

`utils.export_to_pdf_with_xournalpp()` shells out to the `xournalpp` CLI to convert the input file into a temporary PDF. This preserves the original visual layout (drawings, backgrounds, etc.).

### Step 2: HTR Predictions

1. **Parse document** -- `documents.get_document()` decompresses the gzip XML and parses it with BeautifulSoup into a `Document` containing `Page` > `Layer` > `Stroke` objects. A factory function dispatches to `XournalDocument` (`.xoj`) or `XournalppDocument` (`.xopp`).

2. **Render pages** -- Each page is rendered to a 150 DPI grayscale image via matplotlib using the stroke coordinates.

3. **Run HTR** -- The external `htr_pipeline` library processes each image:
    - **Word detection** -- An ONNX model locates word regions (scaled to 40%, with 5px margin).
    - **Line clustering** -- DBSCAN groups detected words into lines (discarding lines with fewer than 2 words).
    - **Text recognition** -- A second ONNX model recognises each word via CTC decoding.

    Output is a dictionary mapping page indices to lists of predictions (text + bounding box coordinates in image pixels).

### Step 3: Embed Text in PDF

`xio.write_predictions_to_PDF()` uses PyMuPDF to add text boxes to the temporary PDF from Step 1. Coordinates are converted from 150 DPI image pixels to 72 DPI PDF points. Text is rendered invisibly (`render_mode=3`) by default, making the PDF searchable without visual clutter. In debug mode (`--show-predictions`), text and bounding boxes are drawn visibly.

## Module Structure

```
xournalpp_htr/
    run_htr.py       # CLI entry point
    shortcuts.py     # Orchestrates Steps 1-3
    documents.py     # .xoj/.xopp parsing (Document ABC, Page, Layer, Stroke)
    models.py        # HTR inference wrapper (compute_predictions)
    utils.py         # Argument parsing, xournalpp CLI export
    xio.py           # PDF I/O (PyMuPDF), example loading (HuggingFace Hub)

scripts/
    demo.py          # Gradio web demo

plugin/
    main.lua         # Xournal++ plugin
    config.lua       # Plugin configuration

external/htr_pipeline/
    HTRPipeline/htr_pipeline/
        __init__.py          # read_page() -- main inference API
        reader/              # ONNX text recognition + CTC decoding
        word_detector/       # ONNX word detection + line clustering
        models/              # Pre-trained ONNX model files
```

## Key Data Structures

**Document model** (from `documents.py`):

- `Document` (ABC) -- holds `pages: list[Page]`, `DPI: int`, renders pages to images
    - `XournalDocument` -- for `.xoj` files
    - `XournalppDocument` -- for `.xopp` files
- `Page` -- `meta_data` (width, height), `background`, `layers: list[Layer]`
- `Layer` -- `strokes: list[Stroke]`
- `Stroke` -- `x: np.array`, `y: np.array`, `meta_data` (color, width, ...)

**Predictions** (from `models.py`):

```python
{
    page_index: [
        {"page_index": int, "text": str,
         "xmin": float, "xmax": float, "ymin": float, "ymax": float},
        ...
    ],
    ...
}
```

Bounding box coordinates are in image pixels at 150 DPI.

## External Dependencies

| Dependency | Purpose |
|---|---|
| `xournalpp` (CLI) | Exports `.xoj`/`.xopp` to PDF (Step 1) |
| `beautifulsoup4` + `lxml` | Parses document XML |
| `matplotlib` | Renders pages to images for HTR |
| `opencv-python` | Image loading and processing |
| `onnxruntime` | Runs word detection and text recognition models |
| `scikit-learn` | DBSCAN clustering for line detection |
| `pymupdf` | Embeds text into PDF (Step 3) |
| `gradio` | Web demo UI (demo only) |
| `huggingface_hub` | Downloads example files (demo only) |
