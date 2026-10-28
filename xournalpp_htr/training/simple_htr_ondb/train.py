"""SimpleHTR-OnDB training entrypoint (step 2; step 1 is ``prepare_data``).

Copy of ``simple_htr/train.py`` that trains on the rendered IAM-OnDB crops with
the writer-based train/val split from the manifest. Requires the
``training-simple-htr-ondb`` extra. Run with::

    uv run --extra training-simple-htr-ondb python -m \\
        xournalpp_htr.training.simple_htr_ondb.train --help
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from xournalpp_htr.training.simple_htr_ondb.config import (
    DataConfig,
    SimpleHTROnDBConfig,
)
from xournalpp_htr.training.simple_htr_ondb.dataset import (
    Words_Dataset,
    custom_collate_fn,
    load_iam_samples,
    load_ondb_samples,
)
from xournalpp_htr.training.simple_htr_ondb.network import (
    SimpleHTRNet,
    compute_ctc_loss,
    greedy_decode,
)
from xournalpp_htr.training.simple_htr_ondb.utils import (
    CustomEncoder,
    get_device,
    get_git_commit_hash,
)

cs = ConfigStore.instance()
cs.store(name="simple_htr_ondb", node=SimpleHTROnDBConfig)

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


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
    data_cfg: DataConfig,
    ondb_path: Path,
    batch_size: int,
    num_workers: int,
    target_height: int,
    target_width: int,
    augment: bool = False,
) -> dict:
    if not (data_cfg.use_ondb or data_cfg.use_iam):
        raise ValueError("Enable at least one of data.use_ondb and data.use_iam.")

    train_parts, val_parts = [], []
    if data_cfg.use_ondb:
        for parts, split in ((train_parts, "train"), (val_parts, "val")):
            parts.append(
                load_ondb_samples(ondb_path, split, target_height, target_width)
            )
    if data_cfg.use_iam:
        iam_root = data_cfg.iam_data_path
        if iam_root is None:
            from xournalpp_htr.xio import load_IAM_DB_dataset

            iam_root = load_IAM_DB_dataset()
            print(f"Resolved IAM-DB dataset from HuggingFace Hub: {iam_root}")
        iam = load_iam_samples(
            Path(iam_root),
            target_height,
            target_width,
            data_cfg.iam_val_fraction,
            data_cfg.iam_split_seed,
        )
        train_parts.append(iam["train"])
        val_parts.append(iam["val"])

    train_dataset = Words_Dataset(train_parts, augment=augment)
    val_dataset = Words_Dataset(val_parts, charset=train_dataset.charset, augment=False)
    for name, parts in (("train", train_parts), ("val", val_parts)):
        print(f"{name}: " + ", ".join(f"{p.source}={len(p.images)}" for p in parts))

    val_only = sorted(
        {c for t in val_dataset.text_cache for c in t} - set(train_dataset.charset)
    )
    if val_only:
        print(f"Characters only in val (always counted as errors): {val_only}")

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=data_cfg.shuffle_data_loader,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    return {
        "train": dataloader_train,
        "val": dataloader_val,
        "charset": train_dataset.charset,
    }


def validate(net, dataloader_val, writer, device, global_step, charset):
    net.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_correct = 0
    total_samples = 0
    blank = len(charset)
    per_source: dict[str, list] = {}  # source -> [cer sum, correct, samples]

    for batch in dataloader_val:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        texts = batch["texts"]

        with torch.no_grad():
            log_probs = net(images)
            loss = compute_ctc_loss(log_probs, targets, target_lengths, blank)
            total_loss += loss.item()

            decoded = greedy_decode(log_probs, charset)
            for pred, gt, source in zip(decoded, texts, batch["sources"], strict=False):
                cer = compute_cer(pred, gt)
                total_cer += cer
                total_samples += 1
                stats = per_source.setdefault(source, [0.0, 0, 0])
                stats[0] += cer
                stats[2] += 1
                if pred == gt:
                    total_correct += 1
                    stats[1] += 1

    avg_loss = total_loss / len(dataloader_val)
    avg_cer = total_cer / max(total_samples, 1)
    word_acc = total_correct / max(total_samples, 1)

    writer.add_scalar("loss/val", avg_loss, global_step)
    writer.add_scalar("cer/val", avg_cer, global_step)
    writer.add_scalar("word_accuracy/val", word_acc, global_step)

    by_source = {
        source: {"cer": cer_sum / n, "word_accuracy": correct / n, "samples": n}
        for source, (cer_sum, correct, n) in per_source.items()
    }
    if len(by_source) > 1:
        for source, m in by_source.items():
            writer.add_scalar(f"cer/val_{source}", m["cer"], global_step)
            writer.add_scalar(
                f"word_accuracy/val_{source}", m["word_accuracy"], global_step
            )

    return avg_cer, word_acc, by_source


def train_epoch(net, optimizer, loader, writer, device, global_step, charset):
    net.train()
    blank = len(charset)
    for i, batch in enumerate(loader):
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        log_probs = net(images)
        loss = compute_ctc_loss(log_probs, targets, target_lengths, blank)
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
    cfg: SimpleHTROnDBConfig,
    dataloader_train,
    dataloader_val,
    charset: list[str],
):
    writer = SummaryWriter(output_path / "summary_writer")

    num_classes = len(charset) + 1  # +1 for CTC blank
    net = SimpleHTRNet(num_classes=num_classes, cfg=cfg.model)
    net.to(device)

    with open(output_path / "charset.json", "w") as f:
        json.dump(charset, f)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.learning_rate)

    global_step = 0
    epoch = 0
    best_val_cer = float("inf")
    patience_counter = 0
    while True:
        epoch += 1
        print(f"Epoch: {epoch}")
        global_step = train_epoch(
            net, optimizer, dataloader_train, writer, device, global_step, charset
        )
        if epoch % cfg.training.val_epoch == 0:
            cer, word_acc, by_source = validate(
                net, dataloader_val, writer, device, global_step, charset
            )
            print(f"  Val CER: {cer:.4f}, Word Accuracy: {word_acc:.4f}")
            if len(by_source) > 1:
                print(
                    "  "
                    + ", ".join(
                        f"{s}: CER {m['cer']:.4f} / acc {m['word_accuracy']:.4f}"
                        for s, m in by_source.items()
                    )
                )

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
                            "per_source": by_source,
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
            "seed_model": cfg.seed.model,
            "dropout": cfg.model.dropout,
            "epoch_max": cfg.training.epoch_max,
            "patience_max": cfg.training.patience_max,
            "augmentation_enabled": cfg.augmentation.enabled,
            "use_ondb": cfg.data.use_ondb,
            "use_iam": cfg.data.use_iam,
        },
        {"best_val_cer": best_val_cer},
    )
    writer.close()


@hydra.main(version_base=None, config_name="simple_htr_ondb")
def main(cfg: SimpleHTROnDBConfig):
    output_path = Path(cfg.output_path)

    ondb_path = Path(cfg.data.data_path) if cfg.data.data_path else DEFAULT_DATA_DIR

    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    device = get_device()

    seed_everything(
        numpy_seed=cfg.seed.model,
        torch_seed=cfg.seed.model,
        random_seed=cfg.seed.model,
    )

    t0 = time.time()
    dataloaders = get_dataloaders(
        data_cfg=cfg.data,
        ondb_path=ondb_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        target_height=cfg.model.input_height,
        target_width=cfg.model.input_width,
        augment=cfg.augmentation.enabled,
    )
    dataloaders_time = time.time() - t0

    charset = dataloaders["charset"]
    print(f"Charset: {len(charset)} characters")

    t0 = time.time()
    train_network(
        output_path=output_path,
        device=device,
        cfg=cfg,
        dataloader_train=dataloaders["train"],
        dataloader_val=dataloaders["val"],
        charset=charset,
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
