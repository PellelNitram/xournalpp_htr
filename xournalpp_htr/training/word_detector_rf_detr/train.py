"""Fine-tune RF-DETR on the IAM word detection dataset.

Downloads the dataset from HuggingFace Hub if not already present, converts
it to the COCO layout RF-DETR expects, and trains an RF-DETR model.
Configuration is managed via Hydra (see ``config.py``).

Usage::

    uv run python -m xournalpp_htr.training.word_detector_rf_detr.train
    uv run python -m xournalpp_htr.training.word_detector_rf_detr.train \\
        training.epochs=100 training.batch_size=8
    uv run python -m xournalpp_htr.training.word_detector_rf_detr.train --cfg job
"""

import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import rfdetr
from huggingface_hub import snapshot_download
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from xournalpp_htr.training.word_detector_rf_detr.config import (
    RESOLUTION_DIVISOR,
    WordDetectorRFDETRConfig,
)

cs = ConfigStore.instance()
cs.store(name="word_detector_rf_detr", node=WordDetectorRFDETRConfig)

SCRIPT_DIR = Path(__file__).resolve().parent

#: RF-DETR consumes Roboflow-style COCO, which names the splits this way.
TRAIN_SPLIT = "train"
VAL_SPLIT = "valid"
ANNOTATION_FILE = "_annotations.coco.json"


def _download_dataset() -> Path:
    return (
        Path(
            snapshot_download(
                repo_id="PellelNitram/xournalpp_htr_IAM_DB",
                repo_type="dataset",
            )
        )
        / "data"
    )


def _parse_xml_gt(xml_dir: Path) -> dict[str, list[dict]]:
    """Read IAM form XML and return per-form word boxes in pixel coordinates."""
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


def _write_coco_split(
    split_dir: Path,
    form_ids: list[str],
    forms: dict[str, list[dict]],
    forms_img_dir: Path,
) -> tuple[int, int]:
    """Copy images into ``split_dir`` and write its COCO annotation file.

    Returns ``(written, skipped)`` form counts.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    skipped = 0

    for image_id, form_id in enumerate(form_ids, start=1):
        img_path = forms_img_dir / f"{form_id}.png"
        if not img_path.exists():
            skipped += 1
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size
        shutil.copy2(img_path, split_dir / f"{form_id}.png")

        images.append(
            {
                "id": image_id,
                "file_name": f"{form_id}.png",
                "width": img_w,
                "height": img_h,
            }
        )

        for box in forms[form_id]:
            # Clamp to the image, as a few IAM components stick out slightly.
            x = max(0.0, min(box["x"], img_w))
            y = max(0.0, min(box["y"], img_h))
            w = max(0.0, min(box["w"], img_w - x))
            h = max(0.0, min(box["h"], img_h - y))
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

    coco = {
        "info": {
            "description": "IAM handwriting word bounding boxes",
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "licenses": [],
        # Roboflow-style COCO reserves category 0 as the supercategory
        # placeholder, so the real classes start at 1.
        "categories": [
            {"id": 0, "name": "words", "supercategory": "none"},
            {"id": 1, "name": "word", "supercategory": "words"},
        ],
        "images": images,
        "annotations": annotations,
    }

    (split_dir / ANNOTATION_FILE).write_text(json.dumps(coco))
    return len(images), skipped


def _prepare_dataset(dataset_dir: Path, val_split: float, seed: int) -> None:
    print("Downloading IAM dataset from HuggingFace Hub...")
    data_dir = _download_dataset()
    print(f"Dataset available at {data_dir}")

    xml_dir = data_dir / "xml"
    forms_img_dir = data_dir / "forms"

    forms = _parse_xml_gt(xml_dir)
    print(
        f"Parsed {sum(len(v) for v in forms.values())} words "
        f"across {len(forms)} forms."
    )

    form_ids = sorted(forms.keys())
    random.seed(seed)
    random.shuffle(form_ids)
    n_val = max(1, int(len(form_ids) * val_split))
    val_ids = form_ids[:n_val]
    train_ids = form_ids[n_val:]

    total_written = 0
    total_skipped = 0
    for split, ids in ((TRAIN_SPLIT, train_ids), (VAL_SPLIT, val_ids)):
        written, skipped = _write_coco_split(
            dataset_dir / split, ids, forms, forms_img_dir
        )
        total_written += written
        total_skipped += skipped
        print(f"  {split}: {written} forms")

    print(
        f"Done: {total_written} forms converted, "
        f"{total_skipped} skipped (image not found)."
    )


def _build_model(cfg: DictConfig):
    """Resolve ``model.variant`` to an ``rfdetr.RFDETR<Variant>`` instance.

    Looked up dynamically so that variants added by newer ``rfdetr`` releases
    (nano, small, medium, ...) work without a code change here.
    """
    if cfg.model.resolution % RESOLUTION_DIVISOR != 0:
        raise ValueError(
            f"model.resolution must be divisible by {RESOLUTION_DIVISOR}, "
            f"got {cfg.model.resolution}."
        )

    class_name = f"RFDETR{cfg.model.variant.capitalize()}"
    if not hasattr(rfdetr, class_name):
        available = sorted(
            name.removeprefix("RFDETR").lower()
            for name in dir(rfdetr)
            if name.startswith("RFDETR") and name != "RFDETR"
        )
        raise ValueError(
            f"Unknown model.variant={cfg.model.variant!r} "
            f"({class_name} not found in rfdetr {rfdetr.__version__}). "
            f"Available: {', '.join(available)}."
        )

    return getattr(rfdetr, class_name)(resolution=cfg.model.resolution)


def _attach_tensorboard(model, log_dir: Path) -> SummaryWriter:
    """Mirror RF-DETR's per-epoch metrics into TensorBoard.

    RF-DETR hands the callback a mapping of metric name to value; anything
    numeric in it is logged as a scalar.
    """
    writer = SummaryWriter(log_dir=str(log_dir))

    def on_fit_epoch_end(log_stats) -> None:
        if not isinstance(log_stats, dict):
            return
        epoch = int(log_stats.get("epoch", 0))
        for key, value in log_stats.items():
            if key == "epoch" or not isinstance(value, (int, float)):
                continue
            writer.add_scalar(key, value, epoch)
        writer.flush()

    model.callbacks["on_fit_epoch_end"].append(on_fit_epoch_end)
    return writer


@hydra.main(version_base=None, config_name="word_detector_rf_detr")
def main(cfg: DictConfig) -> None:
    dataset_dir = (SCRIPT_DIR / cfg.data.dataset_dir).resolve()

    if not (dataset_dir / TRAIN_SPLIT / ANNOTATION_FILE).exists():
        _prepare_dataset(dataset_dir, cfg.data.val_split, cfg.seed.split)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (SCRIPT_DIR / cfg.output_path / f"train_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing run artefacts to {output_dir}")

    model = _build_model(cfg)
    writer = _attach_tensorboard(model, output_dir / "tb")

    try:
        model.train(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            grad_accum_steps=cfg.training.grad_accum_steps,
            lr=cfg.training.lr,
            lr_encoder=cfg.training.lr_encoder,
            weight_decay=cfg.training.weight_decay,
            num_workers=cfg.training.workers,
            device=cfg.training.device,
            checkpoint_interval=cfg.training.checkpoint_interval,
            early_stopping=cfg.training.early_stopping,
            early_stopping_patience=cfg.training.early_stopping_patience,
            tensorboard=False,  # handled by _attach_tensorboard
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
