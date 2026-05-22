"""WordDetector training entrypoint.

Requires the ``training-word-detector`` extra. Run with::

    uv run python -m xournalpp_htr.training.word_detector.train --help
"""

import json
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from xournalpp_htr.training.shared.bounding_box import BoundingBox
from xournalpp_htr.training.shared.postprocessing import (
    MapOrdering,
    binary_classification_metrics,
    cluster_aabbs,
    decode,
    draw_bboxes_on_image,
    fg_by_cc,
    normalize_image_transform,
)
from xournalpp_htr.training.word_detector.config import WordDetectorConfig
from xournalpp_htr.training.word_detector.dataset import (
    IAM_Dataset,
    custom_collate_fn,
    train_augmentation_transform,
)
from xournalpp_htr.training.word_detector.network import (
    WordDetectorNet,
    compute_loss,
)
from xournalpp_htr.training.word_detector.utils import (
    CustomEncoder,
    get_device,
    get_git_commit_hash,
)
from xournalpp_htr.xio import load_IAM_DB_dataset

cs = ConfigStore.instance()
cs.store(name="word_detector", node=WordDetectorConfig)


def seed_everything(numpy_seed=42, torch_seed=1234, random_seed=7):
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(
        f"Seeds set -> random: {random_seed}, numpy: {numpy_seed}, torch: {torch_seed}"
    )


def get_dataloaders(
    data_path: Path,
    percent_train_data: int,
    batch_size: int,
    shuffle_data_loader: bool,
    num_workers: int,
    output_path: Path,
    cache_path: Path,
) -> dict:
    train_transform = train_augmentation_transform
    val_transform = normalize_image_transform

    train_dataset = IAM_Dataset(
        root_dir=data_path,
        input_size=WordDetectorNet.input_size_ImageDimensions,
        output_size=WordDetectorNet.output_size_ImageDimensions,
        force_rebuild_cache=False,
        transform=train_transform,
        cache_path=cache_path,
    )
    val_dataset = IAM_Dataset(
        root_dir=data_path,
        input_size=WordDetectorNet.input_size_ImageDimensions,
        output_size=WordDetectorNet.output_size_ImageDimensions,
        force_rebuild_cache=False,
        transform=val_transform,
        cache_path=cache_path,
    )

    assert len(train_dataset) == len(val_dataset)

    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    split = int(percent_train_data / 100 * len(indices))

    train_indices = indices[:split]
    val_indices = indices[split:]

    with open(output_path / "dataset_split_indices.json", "w") as f:
        json.dump({"train": train_indices, "val": val_indices}, f, indent=4)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_filenames = [sample["filename"] for sample in train_subset]
    val_filenames = [sample["filename"] for sample in val_subset]
    assert len(set(train_filenames + val_filenames)) == len(train_filenames) + len(
        val_filenames
    )

    dataloader_train = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_data_loader,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    return {"train": dataloader_train, "val": dataloader_val}


def validate(
    net,
    dataloader_val,
    writer,
    device,
    input_size,
    output_size,
    global_step,
    cfg_detection,
    regularisation=1e-8,
):
    net.eval()
    avg_loss = 0.0
    tp = fp = fn = 0
    image_counter = 0
    for batch in dataloader_val:
        with torch.no_grad():
            images = batch["images"].to(device)
            gt_encoded = batch["gt_encoded"].to(device)
            y = net(images)
            loss = compute_loss(y, gt_encoded)
            avg_loss += loss.item()
        with torch.no_grad():
            images = batch["images"].to(device)
            gt_encoded = batch["gt_encoded"].to(device)
            y = net(images, apply_softmax=True)
            seg = y[:, MapOrdering.SEG_WORD : MapOrdering.SEG_BACKGROUND + 1, :, :]
            assert seg.min() >= 0.0
            assert seg.max() <= 1.0
        batch_size_here = y.shape[0]
        y = y.to("cpu").numpy()
        for i_element_in_batch in range(batch_size_here):
            y_element = y[i_element_in_batch, :, :, :]
            decoded_aabbs = decode(
                y_element,
                scale=input_size.width / output_size.width,
                comp_fg=fg_by_cc(
                    thres=cfg_detection.fg_threshold,
                    max_num=cfg_detection.max_detections,
                ),
            )
            img_np = batch["images"][i_element_in_batch, 0, :, :].to("cpu").numpy()
            h, w = img_np.shape
            aabbs = [
                aabb.clip(BoundingBox(0, 0, w - 1, h - 1)) for aabb in decoded_aabbs
            ]
            clustered_aabbs = cluster_aabbs(aabbs)
            result = binary_classification_metrics(
                batch["bounding_boxes"][i_element_in_batch], clustered_aabbs
            )
            tp += result["tp"]
            fp += result["fp"]
            fn += result["fn"]
            vis = draw_bboxes_on_image(img_np, clustered_aabbs)
            writer.add_image(
                f"img{image_counter}", vis.transpose((2, 0, 1)), global_step
            )
            image_counter += 1
    avg_loss = avg_loss / len(dataloader_val)
    precision = tp / (tp + fp + regularisation)
    recall = tp / (tp + fn + regularisation)
    f1 = 2 * precision * recall / (precision + recall + regularisation)
    writer.add_scalar("loss/val", avg_loss, global_step)
    writer.add_scalar("f1/val", f1, global_step)
    return f1


def train(net, optimizer, loader, writer, device, global_step):
    net.train()
    for i, loader_item in enumerate(loader):
        images = loader_item["images"].to(device)
        gt_encoded = loader_item["gt_encoded"].to(device)

        optimizer.zero_grad()
        y = net(images)
        loss = compute_loss(y, gt_encoded)
        loss.backward()
        optimizer.step()

        print(f"{i + 1}/{len(loader)}: {loss.item()}")
        writer.add_scalar("loss/train", loss, global_step)
        global_step += 1
    return global_step


def train_network(
    output_path: Path,
    device: str,
    cfg: WordDetectorConfig,
    dataloader_train,
    dataloader_val,
):
    writer = SummaryWriter(output_path / "summary_writer")

    net = WordDetectorNet()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.learning_rate)

    global_step = 0
    epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    while True:
        epoch += 1
        print(f"Epoch: {epoch}")
        global_step = train(
            net, optimizer, dataloader_train, writer, device, global_step
        )
        if epoch % cfg.training.val_epoch == 0:
            f1 = validate(
                net,
                dataloader_val,
                writer,
                device,
                WordDetectorNet.input_size_ImageDimensions,
                WordDetectorNet.output_size_ImageDimensions,
                global_step,
                cfg.detection,
            )

            if f1 > best_val_f1:
                print(
                    f"New best F1 score: {best_val_f1:.4f} -> {f1:.4f}, saving model."
                )
                best_val_f1 = f1
                torch.save(net.state_dict(), output_path / "best_model.pth")
                patience_counter = 0
                with open(output_path / "best_model.json", "w") as f:
                    json.dump(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "f1": best_val_f1,
                        },
                        f,
                        indent=4,
                    )
            else:
                patience_counter += 1

        if patience_counter >= cfg.training.patience_max:
            print("Early stopping triggered.")
            break

        if epoch >= cfg.training.epoch_max:
            print(f"Reached max epoch {cfg.training.epoch_max}, stopping training.")
            break

    writer.add_hparams(
        {
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.batch_size,
            "seed_split": cfg.seed.split,
            "seed_model": cfg.seed.model,
            "patience_max": cfg.training.patience_max,
        },
        {"best_val_f1": best_val_f1},
    )
    writer.close()


@hydra.main(version_base=None, config_name="word_detector")
def main(cfg: WordDetectorConfig):
    output_path = Path(cfg.output_path)

    data_path = Path(cfg.data.data_path) if cfg.data.data_path else None
    cache_path = Path(cfg.data.cache_path)

    if data_path is None and not cache_path.exists():
        data_path = load_IAM_DB_dataset()
        print(f"Resolved IAM-DB dataset from HuggingFace Hub: {data_path}")

    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    device = get_device()

    seed_everything(
        numpy_seed=cfg.seed.split,
        torch_seed=cfg.seed.model,
        random_seed=cfg.seed.model,
    )

    t0 = time.time()
    dataloaders = get_dataloaders(
        data_path=data_path,
        percent_train_data=cfg.data.percent_train_data,
        batch_size=cfg.training.batch_size,
        shuffle_data_loader=cfg.data.shuffle_data_loader,
        num_workers=cfg.training.num_workers,
        output_path=output_path,
        cache_path=cache_path,
    )
    dataloaders_time = time.time() - t0

    t0 = time.time()
    train_network(
        output_path=output_path,
        device=device,
        cfg=cfg,
        dataloader_train=dataloaders["train"],
        dataloader_val=dataloaders["val"],
    )
    training_time = time.time() - t0

    with open(output_path / "git_commit_hash.json", "w") as f:
        json.dump(
            {"git_commit_hash": get_git_commit_hash()}, f, indent=4, cls=CustomEncoder
        )

    with open(output_path / "times.json", "w") as f:
        json.dump(
            {"dataloaders": dataloaders_time, "training": training_time},
            f,
            indent=4,
            cls=CustomEncoder,
        )


if __name__ == "__main__":
    main()
