"""OnlineHTR training entrypoint. Requires the ``training-online-htr`` extra.

Trains the Carbune et al. (2020) BiLSTM-CTC model on IAM-OnDB lines and words
with a writer-based train/val split. Run with::

    uv run --extra training-online-htr python -m \\
        xournalpp_htr.training.online_htr.train --help
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

from xournalpp_htr.training.online_htr.config import OnlineHTRConfig
from xournalpp_htr.training.online_htr.dataset import (
    OnlineHTRDataset,
    build_charset,
    custom_collate_fn,
    load_samples,
)
from xournalpp_htr.training.online_htr.network import (
    OnlineHTRNet,
    compute_ctc_loss,
    greedy_decode,
)
from xournalpp_htr.training.online_htr.utils import (
    CustomEncoder,
    edit_distance,
    get_device,
    get_git_commit_hash,
)

cs = ConfigStore.instance()
cs.store(name="online_htr", node=OnlineHTRConfig)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_cer(predicted: str, target: str) -> float:
    """Character Error Rate via edit distance."""
    if len(target) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    return edit_distance(predicted, target) / len(target)


def get_dataloaders(cfg: OnlineHTRConfig, root: Path) -> dict:
    samples = load_samples(root, cfg.data)
    charset = build_charset(samples["train"])
    for split, items in samples.items():
        counts = {s: sum(i.source == s for i in items) for s in ("line", "word")}
        print(f"{split}: {counts}")

    ppu = cfg.features.points_per_unit
    train = OnlineHTRDataset(samples["train"], charset, ppu, cfg.augmentation)
    val = OnlineHTRDataset(samples["val"], charset, ppu)

    val_only = sorted({c for s in samples["val"] for c in s.text} - set(charset))
    if val_only:
        print(f"Characters only in val (always counted as errors): {val_only}")

    def make(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=cfg.training.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
        )

    return {
        "train": make(train, cfg.data.shuffle_data_loader),
        "val": make(val, False),
        "charset": charset,
    }


def validate(net, loader, writer, device, global_step, charset):
    net.eval()
    blank = len(charset)
    total_loss, total_cer, total_correct, total = 0.0, 0.0, 0, 0
    per_source: dict[str, list] = {}  # source -> [cer sum, correct, samples]

    for batch in loader:
        ink = batch["ink"].to(device)
        lengths = batch["ink_lengths"]
        with torch.no_grad():
            log_probs = net(ink, lengths)
            loss = compute_ctc_loss(
                log_probs,
                batch["targets"].to(device),
                lengths,
                batch["target_lengths"],
                blank,
            )
        total_loss += loss.item()

        decoded = greedy_decode(log_probs, charset, lengths)
        for pred, gt, source in zip(
            decoded, batch["texts"], batch["sources"], strict=True
        ):
            cer = compute_cer(pred, gt)
            stats = per_source.setdefault(source, [0.0, 0, 0])
            stats[0] += cer
            stats[1] += pred == gt
            stats[2] += 1
            total_cer += cer
            total_correct += pred == gt
            total += 1

    avg_loss = total_loss / len(loader)
    avg_cer = total_cer / max(total, 1)
    accuracy = total_correct / max(total, 1)
    writer.add_scalar("loss/val", avg_loss, global_step)
    writer.add_scalar("cer/val", avg_cer, global_step)
    writer.add_scalar("accuracy/val", accuracy, global_step)

    by_source = {
        s: {"cer": c / n, "accuracy": k / n, "samples": n}
        for s, (c, k, n) in per_source.items()
    }
    for s, m in by_source.items():
        writer.add_scalar(f"cer/val_{s}", m["cer"], global_step)
    return avg_cer, accuracy, by_source


def train_epoch(net, optimizer, loader, writer, device, global_step, charset, clip):
    net.train()
    blank = len(charset)
    for i, batch in enumerate(loader):
        lengths = batch["ink_lengths"]
        optimizer.zero_grad()
        log_probs = net(batch["ink"].to(device), lengths)
        loss = compute_ctc_loss(
            log_probs,
            batch["targets"].to(device),
            lengths,
            batch["target_lengths"],
            blank,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip)
        optimizer.step()

        print(f"{i + 1}/{len(loader)}: {loss.item():.4f}")
        writer.add_scalar("loss/train", loss.item(), global_step)
        global_step += 1
    return global_step


def train_network(output_path, device, cfg, loaders, charset):
    writer = SummaryWriter(output_path / "summary_writer")
    net = OnlineHTRNet(num_classes=len(charset) + 1, cfg=cfg.model).to(device)

    with open(output_path / "charset.json", "w") as f:
        json.dump(charset, f)

    t = cfg.training
    optimizer = torch.optim.Adam(net.parameters(), lr=t.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=t.lr_factor, patience=t.lr_patience
    )

    global_step, epoch, patience_counter = 0, 0, 0
    best_val_cer = float("inf")
    while True:
        epoch += 1
        print(f"Epoch: {epoch}")
        global_step = train_epoch(
            net,
            optimizer,
            loaders["train"],
            writer,
            device,
            global_step,
            charset,
            t.grad_clip,
        )
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

        if epoch % t.val_epoch == 0:
            cer, acc, by_source = validate(
                net, loaders["val"], writer, device, global_step, charset
            )
            scheduler.step(cer)
            print(f"  Val CER: {cer:.4f}, accuracy: {acc:.4f}")
            print(
                "  "
                + ", ".join(f"{s}: CER {m['cer']:.4f}" for s, m in by_source.items())
            )
            if cer < best_val_cer:
                print(f"New best CER: {best_val_cer:.4f} -> {cer:.4f}, saving model.")
                best_val_cer, patience_counter = cer, 0
                torch.save(net.state_dict(), output_path / "best_model.pth")
                with open(output_path / "best_model.json", "w") as f:
                    json.dump(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "cer": cer,
                            "accuracy": acc,
                            "per_source": by_source,
                        },
                        f,
                        indent=4,
                    )
            else:
                patience_counter += 1

        if patience_counter >= t.patience_max:
            print("Early stopping triggered.")
            break
        if epoch >= t.epoch_max:
            print(f"Reached max epoch {t.epoch_max}, stopping training.")
            break

    writer.add_hparams(
        {
            "learning_rate": t.learning_rate,
            "batch_size": t.batch_size,
            "seed_model": cfg.seed.model,
            "nodes_per_layer": cfg.model.nodes_per_layer,
            "number_of_layers": cfg.model.number_of_layers,
            "dropout": cfg.model.dropout,
            "augmentation_enabled": cfg.augmentation.enabled,
            "level": cfg.data.level,
        },
        {"best_val_cer": best_val_cer},
    )
    writer.close()


@hydra.main(version_base=None, config_name="online_htr")
def main(cfg: OnlineHTRConfig):
    output_path = Path(cfg.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    if cfg.data.data_path:
        root = Path(cfg.data.data_path)
    else:
        from xournalpp_htr.xio import load_IAM_OnDB_dataset

        root = load_IAM_OnDB_dataset()
        print(f"Resolved IAM-OnDB dataset from HuggingFace Hub: {root}")

    device = get_device()
    seed_everything(cfg.seed.model)

    t0 = time.time()
    loaders = get_dataloaders(cfg, root)
    dataloaders_time = time.time() - t0
    charset = loaders["charset"]
    print(f"Charset: {len(charset)} characters")

    t0 = time.time()
    train_network(output_path, device, cfg, loaders, charset)
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
