# Usage

The usage of the project is fairly simple. First, there is a Python script that performs the actual work & is useful for headless operations like batch processing. Second, and probably much more useful for the average user, the Lua plugin can be used from within Xournal++ and invokes the aforementioned Python script under the hood.

## The Lua plugin

Details relevant for usage of the Lua plugin:

1. Make sure to save your file in Xournal++ beforehand. The plugin will also let you know that you need to save your file first.
2. After installation, navigate to `Plugin > Xournal++ HTR` to invoke the plugin. Then select a filename and press `Save`. Lastly, wait a wee bit until the process is finished; the Xournal++ UI will block while the plugin applies HTR to your file. If you opened Xournal++ through a command-line, you can see progress bars that show the HTR process in real-time.

Note: Currently, the Xournal++ HTR plugin requires you to use a nightly build of Xournal++ because it uses upstream Lua API features that are not yet part of the stable build. Using the officially provided Nightly AppImag, see [here](https://xournalpp.github.io/installation/linux/), is very convenient. The plugin has been tested with the following nightly Linux build of Xournal++:

```
xournalpp 1.2.3+dev (583a4e47)
└──libgtk: 3.24.20
```

## The Python script

It is located in `xournalpp_htr/run_htr.py` and it features a command line interface that documents the usage of the Python script.

The `-p`/`--pipeline` option selects the HTR pipeline. The Lua plugin always
uses the default pipeline.

## Optional: TrOCR pipeline

`2026-10-09_yolo_detector_trocr_large_strip_punct` recognises words with
Microsoft's pretrained
[TrOCR large](https://huggingface.co/microsoft/trocr-large-handwritten) model
instead of our own SimpleHTR model. It is optional and needs extra
dependencies, so it is not part of the default installation.

**Should you use it?** Probably not. On our benchmark it is about as accurate
as our best SimpleHTR pipeline (29.3% vs. 31.7% character error rate, a
difference within noise), but it is much heavier:

- **Download:** about 2.2 GB of model weights on first use, plus PyTorch and
  Hugging Face `transformers` (another 1–2 GB of packages).
- **Speed:** it uses an NVIDIA GPU automatically if one is available. On a CPU
  it is slow: about 3 seconds per word on 4 CPU cores, so several minutes for
  a dense page.

To use it anyway:

1. Install the extra dependencies from the repository root:

    ```bash
    uv sync --extra trocr
    ```

    `uv sync` removes extras that you don't name, so list every extra you use
    (e.g. `uv sync --extra trocr --extra dev`).

2. Select the pipeline with `-p` when running the Python script:

    ```bash
    uv run python xournalpp_htr/run_htr.py \
        -if input.xopp \
        -of output.pdf \
        -p 2026-10-09_yolo_detector_trocr_large_strip_punct
    ```

    The first run downloads the model to the Hugging Face cache
    (`~/.cache/huggingface`); later runs reuse it.

The pipeline has been tested on Linux, with and without a GPU. The background
to this experiment, including benchmark results and design decisions, is on
the [TrOCR model page](models/trocr.md) (issue #156).
