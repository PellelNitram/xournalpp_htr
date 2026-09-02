"""Convert the IAM Handwriting Database to YOLO detection format.

Prerequisites – download these from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
(registration required) and place them in a `raw/` subdirectory:

    raw/
        formsA-D.tgz
        formsE-H.tgz
        formsI-Z.tgz
        words.txt

Usage:
    uv run python prepare_data.py [--raw-dir raw] [--out-dir dataset] [--val-split 0.15]
"""

import argparse
import random
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

from PIL import Image


def extract_forms(raw_dir: Path, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for archive in sorted(raw_dir.glob("forms*.tgz")):
        print(f"Extracting {archive.name} ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp_dir)


def parse_words_txt(words_txt: Path) -> dict[str, list[dict]]:
    """Return {form_id: [{x, y, w, h}, ...]}."""
    forms: dict[str, list[dict]] = defaultdict(list)
    for line in words_txt.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        word_id = parts[0]
        seg_result = parts[1]
        if seg_result == "err":
            continue
        x, y, w, h = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
        # word_id format: a01-000u-00-00 -> form = a01-000u
        segments = word_id.split("-")
        form_id = f"{segments[0]}-{segments[1]}"
        forms[form_id].append({"x": x, "y": y, "w": w, "h": h})
    return forms


def convert_to_yolo(
    forms: dict[str, list[dict]],
    forms_img_dir: Path,
    out_dir: Path,
    val_split: float,
) -> None:
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    form_ids = sorted(forms.keys())
    random.seed(42)
    random.shuffle(form_ids)
    n_val = max(1, int(len(form_ids) * val_split))
    val_ids = set(form_ids[:n_val])

    skipped = 0
    written = 0
    for form_id in form_ids:
        img_path = forms_img_dir / f"{form_id}.png"
        if not img_path.exists():
            skipped += 1
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        split = "val" if form_id in val_ids else "train"
        dst_img = out_dir / "images" / split / f"{form_id}.png"
        shutil.copy2(img_path, dst_img)

        label_lines = []
        for box in forms[form_id]:
            x_center = (box["x"] + box["w"] / 2) / img_w
            y_center = (box["y"] + box["h"] / 2) / img_h
            bw = box["w"] / img_w
            bh = box["h"] / img_h
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            bw = min(bw, 1.0)
            bh = min(bh, 1.0)
            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_dir / "labels" / split / f"{form_id}.txt"
        label_path.write_text("\n".join(label_lines) + "\n")
        written += 1

    print(f"Done: {written} forms converted, {skipped} skipped (image not found).")
    n_train = written - len(
        [f for f in val_ids if (out_dir / "labels" / "val" / f"{f}.txt").exists()]
    )
    n_val_actual = written - n_train
    print(f"  train: {n_train}  |  val: {n_val_actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IAM to YOLO format")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    words_txt = args.raw_dir / "words.txt"
    if not words_txt.exists():
        raise FileNotFoundError(
            f"{words_txt} not found. Download it from "
            "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"
        )

    tmp_forms = Path("_tmp_forms")
    extract_forms(args.raw_dir, tmp_forms)

    forms_img_dir = tmp_forms / "forms"
    if not forms_img_dir.exists():
        candidates = list(tmp_forms.rglob("*.png"))
        if candidates:
            forms_img_dir = candidates[0].parent
        else:
            raise FileNotFoundError("No form images found after extraction.")

    forms = parse_words_txt(words_txt)
    print(
        f"Parsed {sum(len(v) for v in forms.values())} words across {len(forms)} forms."
    )
    convert_to_yolo(forms, forms_img_dir, args.out_dir, args.val_split)

    shutil.rmtree(tmp_forms, ignore_errors=True)
    print("Cleaned up temp files.")


if __name__ == "__main__":
    main()
