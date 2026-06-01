"""SimpleHTR training entrypoint.

Requires the ``training-simple-htr`` extra. Run with::

    uv run python -m xournalpp_htr.training.simple_htr.train --help
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

from xournalpp_htr.training.simple_htr.config import SimpleHTRConfig
from xournalpp_htr.training.simple_htr.dataset import (
    IAM_Words_Dataset,
    custom_collate_fn,
)
from xournalpp_htr.training.simple_htr.network import (
    SimpleHTRNet,
    compute_ctc_loss,
    greedy_decode,
)
from xournalpp_htr.training.simple_htr.utils import (
    CustomEncoder,
    get_device,
    get_git_commit_hash,
)
from xournalpp_htr.xio import load_IAM_DB_dataset

cs = ConfigStore.instance()
cs.store(name="simple_htr", node=SimpleHTRConfig)


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


def compute_cer(predicted: str, target: str) -> float:
    """Character Error Rate via edit distance."""
    if len(target) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    d = np.zeros((len(predicted) + 1, len(target) + 1), dtype=np.int32)
    for i in range(len(predicted) + 1):
        d[i, 0] = i
    for j in range(len(target) + 1):
        d[0, j] = j
    for i in range(1, len(predicted) + 1):
        for j in range(1, len(target) + 1):
            cost = 0 if predicted[i - 1] == target[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)

    return d[len(predicted), len(target)] / len(target)


def get_dataloaders(
    data_path: Path,
    percent_train_data: int,
    batch_size: int,
    shuffle_data_loader: bool,
    num_workers: int,
    output_path: Path,
    cache_path: Path,
    target_height: int,
    target_width: int,
    augment: bool = False,
) -> dict:
    train_dataset = IAM_Words_Dataset(
        root_dir=data_path,
        target_height=target_height,
        target_width=target_width,
        force_rebuild_cache=False,
        augment=augment,
        cache_path=cache_path,
    )
    val_dataset = IAM_Words_Dataset(
        root_dir=data_path,
        target_height=target_height,
        target_width=target_width,
        force_rebuild_cache=False,
        augment=False,
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


def validate(net, dataloader_val, writer, device, global_step):
    net.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader_val:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        texts = batch["texts"]

        with torch.no_grad():
            log_probs = net(images)
            loss = compute_ctc_loss(log_probs, targets, target_lengths)
            total_loss += loss.item()

            decoded = greedy_decode(log_probs)
            for pred, gt in zip(decoded, texts, strict=False):
                total_cer += compute_cer(pred, gt)
                if pred == gt:
                    total_correct += 1
                total_samples += 1

    avg_loss = total_loss / len(dataloader_val)
    avg_cer = total_cer / max(total_samples, 1)
    word_acc = total_correct / max(total_samples, 1)

    writer.add_scalar("loss/val", avg_loss, global_step)
    writer.add_scalar("cer/val", avg_cer, global_step)
    writer.add_scalar("word_accuracy/val", word_acc, global_step)

    return avg_cer, word_acc


def train_epoch(net, optimizer, loader, writer, device, global_step):
    net.train()
    for i, batch in enumerate(loader):
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        log_probs = net(images)
        loss = compute_ctc_loss(log_probs, targets, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

        print(f"{i + 1}/{len(loader)}: {loss.item():.4f}")
        writer.add_scalar("loss/train", loss.item(), global_step)
        global_step += 1
    return global_step


def train_network(
    output_path: Path,
    device: str,
    cfg: SimpleHTRConfig,
    dataloader_train,
    dataloader_val,
):
    writer = SummaryWriter(output_path / "summary_writer")

    net = SimpleHTRNet()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.learning_rate)

    global_step = 0
    epoch = 0
    best_val_cer = float("inf")
    patience_counter = 0
    while True:
        epoch += 1
        print(f"Epoch: {epoch}")
        global_step = train_epoch(
            net, optimizer, dataloader_train, writer, device, global_step
        )
        if epoch % cfg.training.val_epoch == 0:
            cer, word_acc = validate(net, dataloader_val, writer, device, global_step)
            print(f"  Val CER: {cer:.4f}, Word Accuracy: {word_acc:.4f}")

            if cer < best_val_cer:
                print(f"New best CER: {best_val_cer:.4f} -> {cer:.4f}, saving model.")
                best_val_cer = cer
                torch.save(net.state_dict(), output_path / "best_model.pth")
                patience_counter = 0
                with open(output_path / "best_model.json", "w") as f:
                    json.dump(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "cer": best_val_cer,
                            "word_accuracy": word_acc,
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
            "augmentation_enabled": cfg.augmentation.enabled,
        },
        {"best_val_cer": best_val_cer},
    )
    writer.close()


@hydra.main(version_base=None, config_name="simple_htr")
def main(cfg: SimpleHTRConfig):
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
        target_height=cfg.model.input_height,
        target_width=cfg.model.input_width,
        augment=cfg.augmentation.enabled,
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
