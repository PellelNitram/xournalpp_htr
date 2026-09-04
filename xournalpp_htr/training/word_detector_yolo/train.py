"""Fine-tune YOLOv8 on the IAM word detection dataset.

Downloads the dataset from HuggingFace Hub if not already present, converts
it to YOLO format, and trains a YOLOv8 model. Configuration is managed via
Hydra (see ``config.py``).

Usage::

    uv run python -m xournalpp_htr.training.word_detector_yolo.train
    uv run python -m xournalpp_htr.training.word_detector_yolo.train \\
        training.epochs=100 training.batch=8 training.device=0
    uv run python -m xournalpp_htr.training.word_detector_yolo.train --cfg job
"""

import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import numpy as np
from huggingface_hub import snapshot_download
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from xournalpp_htr.training.word_detector_yolo.config import WordDetectorYOLOConfig

cs = ConfigStore.instance()
cs.store(name="word_detector_yolo", node=WordDetectorYOLOConfig)

SCRIPT_DIR = Path(__file__).resolve().parent
N_PREVIEW_IMAGES = 3
PREVIEW_SEED = 42


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

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    form_ids = sorted(forms.keys())
    random.seed(seed)
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
        dst_img = dataset_dir / "images" / split / f"{form_id}.png"
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

        label_path = dataset_dir / "labels" / split / f"{form_id}.txt"
        label_path.write_text("\n".join(label_lines) + "\n")
        written += 1

    print(f"Done: {written} forms converted, {skipped} skipped (image not found).")

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: word\n"
    )
    print(f"Wrote {data_yaml}")


def _select_preview_images(data_yaml: Path) -> list[Path]:
    import yaml

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    val_dir = Path(cfg["path"]) / cfg["val"]
    images = sorted(val_dir.glob("*.png")) + sorted(val_dir.glob("*.jpg"))
    rng = random.Random(PREVIEW_SEED)
    return rng.sample(images, min(N_PREVIEW_IMAGES, len(images)))


def _log_predictions(trainer):
    if not hasattr(trainer, "_tb_writer"):
        return
    writer = trainer._tb_writer
    preview_imgs = trainer._preview_images
    epoch = trainer.epoch + 1
    checkpoint = Path(trainer.save_dir) / "weights" / "last.pt"
    if not checkpoint.exists():
        return
    pred_model = YOLO(str(checkpoint))
    for img_path in preview_imgs:
        img = cv2.imread(str(img_path))
        results = pred_model.predict(img, conf=0.25, imgsz=1024, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_tensor = np.transpose(annotated_rgb, (2, 0, 1))
        writer.add_image(f"predictions/{img_path.stem}", img_tensor, epoch)
    writer.flush()


@hydra.main(version_base=None, config_name="word_detector_yolo")
def main(cfg: DictConfig) -> None:
    dataset_dir = (SCRIPT_DIR / cfg.data.dataset_dir).resolve()
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        _prepare_dataset(dataset_dir, cfg.data.val_split, cfg.seed.split)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"train_{timestamp}"

    preview_images = _select_preview_images(data_yaml)
    print(f"Preview images for TensorBoard: {[p.name for p in preview_images]}")

    model = YOLO(cfg.model.variant)

    def on_train_start(trainer):
        log_dir = Path(trainer.save_dir)
        trainer._tb_writer = SummaryWriter(log_dir=str(log_dir))
        trainer._preview_images = preview_images

    def on_fit_epoch_end(trainer):
        _log_predictions(trainer)

    def on_train_end(trainer):
        if hasattr(trainer, "_tb_writer"):
            trainer._tb_writer.close()

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    model.train(
        data=str(data_yaml),
        name=name,
        epochs=cfg.training.epochs,
        batch=cfg.training.batch,
        imgsz=cfg.model.imgsz,
        device=cfg.training.device,
        workers=cfg.training.workers,
        patience=cfg.training.patience,
        save=True,
        save_period=cfg.training.save_period,
        pretrained=True,
        optimizer=cfg.training.optimizer,
        lr0=cfg.training.lr0,
        lrf=cfg.training.lrf,
        warmup_epochs=cfg.training.warmup_epochs,
        mosaic=cfg.augmentation.mosaic,
        hsv_h=cfg.augmentation.hsv_h,
        hsv_s=cfg.augmentation.hsv_s,
        hsv_v=cfg.augmentation.hsv_v,
        degrees=cfg.augmentation.degrees,
        translate=cfg.augmentation.translate,
        scale=cfg.augmentation.scale,
        fliplr=cfg.augmentation.fliplr,
        flipud=cfg.augmentation.flipud,
        project=str(SCRIPT_DIR / "runs" / "detect"),
    )


if __name__ == "__main__":
    main()
