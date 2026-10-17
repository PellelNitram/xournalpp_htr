"""Render IAM-OnDB into word-level crops for training SimpleHTR on online data.

Step 1 of 2 (step 2 is training). Run with::

    uv run python -m xournalpp_htr.training.simple_htr_ondb.prepare_data \\
        --output-dir data/simple_htr_ondb

Pipeline per IAM-OnDB line:

1. Glue standalone punctuation tokens onto words, then cut the strokes into
   words at the largest horizontal gaps (``word_splitting``). Lines whose gaps
   don't match the transcription are dropped.
2. Scale the line so its median word height is sampled log-uniformly from the
   range measured on the benchmark documents (a size statistic only).
3. Render the line with ``StrokesDocument.save_page_as_image`` at the DPI the
   inference pipeline uses, to a JPEG, and read it back as grayscale, exactly
   like ``compute_predictions`` does.
4. Crop every word at its tight stroke box plus a small margin.

Train/val are split by writer. Outputs, all under ``--output-dir``:
``images/<split>/*.png``, ``manifest.jsonl``, ``stats.json``,
``preview_crops.png`` and ``preview_lines.png`` to eyeball the result.
"""

import argparse
import json
import math
import os
import tempfile
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from xournalpp_htr.documents import StrokesDocument  # noqa: E402
from xournalpp_htr.training.data.datasets import IAM_OnDB_Dataset  # noqa: E402
from xournalpp_htr.training.simple_htr_ondb.word_splitting import (  # noqa: E402
    group_tokens,
    split_into_words,
)
from xournalpp_htr.xio import load_IAM_OnDB_dataset  # noqa: E402

DOC_DPI = 72  # unit of StrokesDocument coordinates (points)


@dataclass
class PrepareConfig:
    render_dpi: float = 150.0  # same as RENDER_DPI in `compute_predictions`
    # Median word height per line, in points. p5/p95 of the benchmark's
    # ground-truth word boxes (median 21.5 pt); sampled log-uniformly.
    word_height_pt_range: tuple[float, float] = (10.0, 46.0)
    crop_margin_px: int = 4
    page_padding_pt: float = 20.0
    min_gap_ratio: float = 1.5
    val_writer_fraction: float = 0.1
    seed: int = 42


def list_line_ids(root: Path) -> list[str]:
    """All IAM-OnDB line ids with a transcription, sorted."""
    ids = []
    for _, _, files in os.walk(root / "lineStrokes-all"):
        for f in files:
            if f.endswith(".xml"):
                name = f[: -len(".xml")]
                if name not in IAM_OnDB_Dataset.SAMPLES_NOT_TO_STORE:
                    ids.append(name)
    return sorted(ids)


def load_form_writers(root: Path) -> dict[str, str]:
    """Map form id (``a01-000u``) to writer id from ``forms.txt``."""
    mapping = {}
    for line in (root / "forms.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            form, writer, *_ = line.split()
            mapping[form] = writer
    return mapping


def form_of(line_id: str) -> str:
    """``a01-000u-01`` -> ``a01-000u``."""
    return line_id.rsplit("-", 1)[0]


def assign_splits(writers: list[str], val_fraction: float, seed: int) -> dict[str, str]:
    """Assign whole writers to ``train`` or ``val``."""
    unique = sorted(set(writers))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, round(len(unique) * val_fraction))
    return {w: ("val" if i < n_val else "train") for i, w in enumerate(unique)}


@dataclass
class RenderedLine:
    image: np.ndarray  # grayscale render of the whole line
    words: list[str]
    boxes: list[tuple[int, int, int, int]]  # crop boxes (x0, y0, x1, y1) in px
    scale: float  # points per raw OnDB unit
    word_height_pt: float  # sampled median word height


def render_line(
    line_id: str, root: Path, cfg: PrepareConfig, tmp_dir: Path
) -> tuple[RenderedLine | None, str | None]:
    """Split, scale and render one line. Returns ``(line, reject_reason)``."""
    df, text = IAM_OnDB_Dataset.load_IAM_OnDB_sample(line_id, root)
    df["y"] *= -1  # the loader flips y upwards; documents have y pointing down

    words, reason = group_tokens(text)
    if reason:
        return None, reason

    strokes = [
        (g["x"].to_numpy(), g["y"].to_numpy()) for _, g in df.groupby("stroke_nr")
    ]
    extents = np.array([[x.min(), x.max()] for x, _ in strokes])
    groups, reason = split_into_words(extents, len(words), cfg.min_gap_ratio)
    if reason:
        return None, reason

    # Deterministic per line, independent of processing order.
    rng = np.random.default_rng([cfg.seed, zlib.crc32(line_id.encode())])
    lo, hi = cfg.word_height_pt_range
    word_height_pt = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))

    raw_heights = [np.ptp(np.concatenate([strokes[i][1] for i in g])) for g in groups]
    median_raw = float(np.median(raw_heights))
    if median_raw <= 0:
        return None, "degenerate_height"
    scale = word_height_pt / median_raw

    x_min = min(x.min() for x, _ in strokes)
    y_min = min(y.min() for _, y in strokes)
    pad = cfg.page_padding_pt
    scaled = [
        ((x - x_min) * scale + pad, (y - y_min) * scale + pad) for x, y in strokes
    ]
    width = max(x.max() for x, _ in scaled) + pad
    height = max(y.max() for _, y in scaled) + pad

    # The inference pipeline renders to a temp JPEG and reads it as grayscale.
    out_file = tmp_dir / f"{line_id}.jpg"
    document = StrokesDocument(scaled, width, height)
    document.save_page_as_image(0, out_file, False, dpi=cfg.render_dpi)
    image = cv2.imread(str(out_file), cv2.IMREAD_GRAYSCALE)
    out_file.unlink()

    px_per_pt = cfg.render_dpi / DOC_DPI
    boxes = []
    for g in groups:
        xs = np.concatenate([scaled[i][0] for i in g]) * px_per_pt
        ys = np.concatenate([scaled[i][1] for i in g]) * px_per_pt
        m = cfg.crop_margin_px
        boxes.append(
            (
                max(0, int(xs.min()) - m),
                max(0, int(ys.min()) - m),
                min(image.shape[1], int(math.ceil(xs.max())) + m),
                min(image.shape[0], int(math.ceil(ys.max())) + m),
            )
        )
    return RenderedLine(image, words, boxes, scale, word_height_pt), None


def _process(args) -> tuple[str, list[dict], str | None]:
    """Worker: render one line and write its crops. Returns manifest rows."""
    line_id, root, cfg, split, writer, out_dir, tmp_dir = args
    try:
        line, reason = render_line(line_id, root, cfg, tmp_dir)
    except Exception as e:  # noqa: BLE001 - report instead of crashing the pool
        return line_id, [], f"error:{type(e).__name__}"
    if line is None:
        return line_id, [], reason

    rows = []
    for k, (word, (x0, y0, x1, y1)) in enumerate(
        zip(line.words, line.boxes, strict=False)
    ):
        crop = line.image[y0:y1, x0:x1]
        if crop.size == 0:
            return line_id, [], "empty_crop"
        rel = Path("images") / split / f"{line_id}_{k:02d}.png"
        cv2.imwrite(str(out_dir / rel), crop)
        rows.append(
            {
                "image": str(rel),
                "text": word,
                "split": split,
                "writer": writer,
                "line_id": line_id,
                "word_index": k,
                "scale": line.scale,
                "word_height_pt": line.word_height_pt,
                "height_px": int(crop.shape[0]),
                "width_px": int(crop.shape[1]),
            }
        )
    return line_id, rows, None


def write_previews(
    rows: list[dict], root: Path, cfg: PrepareConfig, out_dir: Path, tmp_dir: Path
) -> None:
    """Contact sheets: random word crops, and whole lines with their word boxes."""
    rng = np.random.default_rng(cfg.seed)

    picks = rng.choice(len(rows), size=min(48, len(rows)), replace=False)
    fig, axes = plt.subplots(8, 6, figsize=(14, 12))
    for ax, i in zip(axes.flat, picks, strict=False):
        ax.imshow(cv2.imread(str(out_dir / rows[i]["image"]), 0), cmap="gray")
        ax.set_title(rows[i]["text"], fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(picks) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "preview_crops.png", dpi=110)
    plt.close(fig)

    line_ids = sorted({r["line_id"] for r in rows})
    chosen = rng.choice(len(line_ids), size=min(6, len(line_ids)), replace=False)
    fig, axes = plt.subplots(len(chosen), 1, figsize=(14, 2.6 * len(chosen)))
    for ax, i in zip(np.atleast_1d(axes), chosen, strict=False):
        line, _ = render_line(line_ids[i], root, cfg, tmp_dir)
        ax.imshow(line.image, cmap="gray")
        for word, (x0, y0, x1, y1) in zip(line.words, line.boxes, strict=False):
            ax.add_patch(
                plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1)
            )
            ax.text(x0, y0 - 2, word, color="red", fontsize=8)
        ax.set_title(
            f"{line_ids[i]} (word height {line.word_height_pt:.0f} pt)", fontsize=8
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "preview_lines.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only process N lines.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    cfg = PrepareConfig()
    out_dir: Path = args.output_dir
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    root = load_IAM_OnDB_dataset()
    form_writers = load_form_writers(root)
    line_ids = list_line_ids(root)
    if args.limit:
        line_ids = line_ids[: args.limit]

    # A few lines belong to forms that forms.txt doesn't list; no writer, no split.
    no_writer = [i for i in line_ids if form_of(i) not in form_writers]
    line_ids = [i for i in line_ids if form_of(i) in form_writers]

    writer_split = assign_splits(
        [form_writers[form_of(i)] for i in line_ids], cfg.val_writer_fraction, cfg.seed
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        jobs = []
        for line_id in line_ids:
            writer = form_writers[form_of(line_id)]
            jobs.append(
                (line_id, root, cfg, writer_split[writer], writer, out_dir, tmp_dir)
            )

        rows: list[dict] = []
        rejected: dict[str, list[str]] = {"no_writer": no_writer} if no_writer else {}
        with Pool(args.workers) as pool:
            for line_id, line_rows, reason in tqdm(
                pool.imap_unordered(_process, jobs, chunksize=16),
                total=len(jobs),
                desc="Render lines",
            ):
                if reason:
                    rejected.setdefault(reason, []).append(line_id)
                rows.extend(line_rows)

        rows.sort(key=lambda r: (r["line_id"], r["word_index"]))
        with open(out_dir / "manifest.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        by_split = Counter(r["split"] for r in rows)
        writers_by_split = {
            s: len({r["writer"] for r in rows if r["split"] == s}) for s in by_split
        }
        stats = {
            "config": asdict(cfg),
            "lines_total": len(line_ids) + len(no_writer),
            "lines_accepted": len({r["line_id"] for r in rows}),
            "lines_rejected_by_reason": {
                k: len(v) for k, v in sorted(rejected.items())
            },
            "rejected_examples": {
                k: sorted(v)[:10] for k, v in sorted(rejected.items())
            },
            "words_by_split": dict(by_split),
            "writers_by_split": writers_by_split,
            "charset": "".join(sorted({c for r in rows for c in r["text"]})),
            "crop_height_px_percentiles_5_50_95": np.percentile(
                [r["height_px"] for r in rows], [5, 50, 95]
            )
            .round(1)
            .tolist(),
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        print(
            json.dumps(
                {k: v for k, v in stats.items() if k != "rejected_examples"}, indent=2
            )
        )

        write_previews(rows, root, cfg, out_dir, tmp_dir)


if __name__ == "__main__":
    main()
