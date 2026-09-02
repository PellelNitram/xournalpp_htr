"""Convert the IAM Handwriting Database to YOLO detection format.

Downloads the dataset from HuggingFace Hub (PellelNitram/xournalpp_htr_IAM_DB)
and converts form images + XML ground truth into YOLO-format labels.

Requires a valid HuggingFace token with access to the private repository,
set via the ``HF_TOKEN`` environment variable or ``huggingface-cli login``.

Usage:
    uv run python prepare_data.py [--out-dir dataset] [--val-split 0.15]
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image


def download_dataset() -> Path:
    """Download IAM-DB from HuggingFace Hub, return path to data/ folder."""
    return (
        Path(
            snapshot_download(
                repo_id="PellelNitram/xournalpp_htr_IAM_DB",
                repo_type="dataset",
            )
        )
        / "data"
    )


def parse_xml_gt(xml_dir: Path) -> dict[str, list[dict]]:
    """Return {form_id: [{x, y, w, h}, ...]} from XML ground truth files."""
    forms: dict[str, list[dict]] = defaultdict(list)
    for xml_path in sorted(xml_dir.glob("*.xml")):
        form_id = xml_path.stem
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for line in root.findall("./handwritten-part/line"):
            for word in line.findall("./word"):
                components = word.findall("./cmp")
                if not components:
                    continue
                x_min, x_max = float("inf"), 0
                y_min, y_max = float("inf"), 0
                for cmp in components:
                    x = float(cmp.attrib["x"])
                    y = float(cmp.attrib["y"])
                    w = float(cmp.attrib["width"])
                    h = float(cmp.attrib["height"])
                    x_min = min(x_min, x)
                    x_max = max(x_max, x + w)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y + h)
                forms[form_id].append(
                    {
                        "x": x_min,
                        "y": y_min,
                        "w": x_max - x_min,
                        "h": y_max - y_min,
                    }
                )
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
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    print("Downloading IAM dataset from HuggingFace Hub...")
    data_dir = download_dataset()
    print(f"Dataset available at {data_dir}")

    xml_dir = data_dir / "xml"
    forms_img_dir = data_dir / "forms"

    forms = parse_xml_gt(xml_dir)
    print(
        f"Parsed {sum(len(v) for v in forms.values())} words across {len(forms)} forms."
    )
    convert_to_yolo(forms, forms_img_dir, args.out_dir, args.val_split)


if __name__ == "__main__":
    main()
