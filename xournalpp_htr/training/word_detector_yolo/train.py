"""Fine-tune YOLOv8s on the IAM word detection dataset.

Usage:
    uv run python train.py [--epochs 50] [--batch 16] [--imgsz 1024] [--device 0]

The trained model is saved under runs/detect/train_<timestamp>/weights/best.pt
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

N_PREVIEW_IMAGES = 3
PREVIEW_SEED = 42


def _select_preview_images(data_yaml: str) -> list[Path]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8s for word detection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"train_{timestamp}"

    preview_images = _select_preview_images("data.yaml")
    print(f"Preview images for TensorBoard: {[p.name for p in preview_images]}")

    model = YOLO("yolov8s.pt")

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
        data="data.yaml",
        name=name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=8,
        patience=10,
        save=True,
        save_period=5,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        mosaic=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=2.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.0,
        flipud=0.0,
        project="runs/detect",
    )


if __name__ == "__main__":
    main()
