"""WordDetector training entrypoint.

Requires the ``training-word-detector`` extra. Run with::

    uv run python -m xournalpp_htr.training.word_detector.train --help
"""

import argparse
import json
import random
import time
from datetime import datetime  # noqa: F401  (kept for parity with logs)
from pathlib import Path

import numpy as np
import torch
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
from xournalpp_htr.training.word_detector.dataset import (
    IAM_Dataset,
    custom_collate_fn,
)
from xournalpp_htr.training.word_detector.network import (
    WordDetectorNet,
    compute_loss,
)
from xournalpp_htr.training.word_detector.utils import (
    CustomEncoder,
    get_git_commit_hash,
)
from xournalpp_htr.xio import load_IAM_DB_dataset

global_step = 0  # TODO: Make global step non-global as it's very bad practise.


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Train a WordDetectorNet model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    training_group = parser.add_argument_group("Training Settings")
    training_group.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    training_group.add_argument(
        "--val_epoch", type=int, default=1, help="Validation frequency in epochs"
    )
    training_group.add_argument(
        "--epoch_max", type=int, default=3, help="Maximum number of epochs"
    )
    training_group.add_argument(
        "--patience_max", type=int, default=50, help="Early stopping patience"
    )
    training_group.add_argument("--batch_size", type=int, default=32, help="Batch size")
    training_group.add_argument(
        "--num_workers", type=int, default=1, help="Number of data loader workers"
    )

    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help=(
            "Path to the IAM-DB dataset root (the directory containing "
            "'forms/' and 'xml/'). If omitted, the dataset is resolved from "
            "the HuggingFace Hub via load_IAM_DB_dataset() (cached locally)."
        ),
    )
    data_group.add_argument(
        "--percent_train_data",
        type=int,
        default=80,
        help="Percentage of data used for training",
    )
    data_group.add_argument(
        "--no-shuffle-data-loader",
        dest="shuffle_data_loader",
        action="store_false",
        help="Disable shuffling of data loader (default: enabled)",
    )
    parser.set_defaults(shuffle_data_loader=True)
    data_group.add_argument(
        "--cache_path",
        type=Path,
        default=Path.home() / "dataset_cache.pickle",
        help="Path to dataset cache file",
    )

    seed_group = parser.add_argument_group("Reproducibility Settings")
    seed_group.add_argument(
        "--seed_split", type=int, default=42, help="Random seed for dataset splitting"
    )
    seed_group.add_argument(
        "--seed_model",
        type=int,
        default=1337,
        help="Random seed for model initialization",
    )

    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "--output_path",
        type=Path,
        default=Path("test_output_path"),
        help="Output directory (created if it doesn't exist)",
    )

    return vars(parser.parse_args())


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
    train_transform = normalize_image_transform
    val_transform = normalize_image_transform
    # TODO: Implement augmentations, changing at every batch.

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
    net, dataloader_val, writer, device, input_size, output_size, regularisation=1e-8
):
    global global_step
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
                comp_fg=fg_by_cc(thres=0.5, max_num=1000),
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


def train(net, optimizer, loader, writer, device):
    global global_step
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


def train_network(
    output_path: Path,
    device: str,
    learning_rate: float,
    batch_size: int,
    dataloader_train,
    dataloader_val,
    epoch_max: int,
    patience_max: int,
    val_epoch: int,
    seed_split: int,
    seed_model: int,
):
    writer = SummaryWriter(output_path / "summary_writer")

    net = WordDetectorNet()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    while True:
        epoch += 1
        print(f"Epoch: {epoch}")
        train(net, optimizer, dataloader_train, writer, device)
        if epoch % val_epoch == 0:
            f1 = validate(
                net,
                dataloader_val,
                writer,
                device,
                WordDetectorNet.input_size_ImageDimensions,
                WordDetectorNet.output_size_ImageDimensions,
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

        if patience_counter >= patience_max:
            print("Early stopping triggered.")
            break

        if epoch >= epoch_max:
            print(f"Reached max epoch {epoch_max}, stopping training.")
            break

    writer.add_hparams(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed_split": seed_split,
            "seed_model": seed_model,
            "patience_max": patience_max,
        },
        {"best_val_f1": best_val_f1},
    )
    writer.close()


def main(args: dict):
    if args["data_path"] is None:
        args["data_path"] = load_IAM_DB_dataset()
        print(f"Resolved IAM-DB dataset from HuggingFace Hub: {args['data_path']}")

    args["output_path"].mkdir(exist_ok=True, parents=True)

    with open(args["output_path"] / "args.json", "w") as f:
        json.dump(args, f, indent=4, cls=CustomEncoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(
        numpy_seed=args["seed_split"],
        torch_seed=args["seed_model"],
        random_seed=args["seed_model"],
    )

    t0 = time.time()
    dataloaders = get_dataloaders(
        data_path=args["data_path"],
        percent_train_data=args["percent_train_data"],
        batch_size=args["batch_size"],
        shuffle_data_loader=args["shuffle_data_loader"],
        num_workers=args["num_workers"],
        output_path=args["output_path"],
        cache_path=args["cache_path"],
    )
    dataloaders_time = time.time() - t0

    t0 = time.time()
    train_network(
        output_path=args["output_path"],
        device=device,
        learning_rate=args["learning_rate"],
        batch_size=args["batch_size"],
        dataloader_train=dataloaders["train"],
        dataloader_val=dataloaders["val"],
        epoch_max=args["epoch_max"],
        patience_max=args["patience_max"],
        val_epoch=args["val_epoch"],
        seed_split=args["seed_split"],
        seed_model=args["seed_model"],
    )
    training_time = time.time() - t0

    with open(args["output_path"] / "git_commit_hash.json", "w") as f:
        json.dump(
            {"git_commit_hash": get_git_commit_hash()}, f, indent=4, cls=CustomEncoder
        )

    with open(args["output_path"] / "times.json", "w") as f:
        json.dump(
            {"dataloaders": dataloaders_time, "training": training_time},
            f,
            indent=4,
            cls=CustomEncoder,
        )


if __name__ == "__main__":
    main(parse_args())
