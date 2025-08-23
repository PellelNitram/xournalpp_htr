import argparse
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from my_code import IAM_Dataset
from my_code import ImageDimensions
from my_code import custom_collate_fn
from my_code import count_parameters
from my_code import fg_by_cc
from my_code import cluster_aabbs
from my_code import binary_classification_metrics
from my_code import draw_bboxes_on_image
from my_code import MapOrdering
from my_code import encode, decode, BoundingBox, ImageDimensions
from my_code import ModifiedResNet18
from my_code import WordDetectorNet
from my_code import compute_loss
from my_code import normalize_image_transform
from my_code import CustomEncoder


global_step = 0 # Fix this and remove global step property!

def parse_args() -> dict:
    return {
        'learning_rate': 0.001,
        'val_epoch': 1,
        'epoch_max': 3,
        'patience_max': 50,
        'data_path': Path.home() / 'Development/WordDetectorNN/data/train',
        'percent_train_data': 80,
        'shuffle_data_loader': True,
        'batch_size': 32,
        'num_workers': 1,
        'output_path': Path('test_output_path'), # Doesn't have to exist bc it's created
        'seed_split': 42,
        'seed_model': 1337,
    }

def seed_everything(numpy_seed=42, torch_seed=1234, random_seed=7):
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set -> random: {random_seed}, numpy: {numpy_seed}, torch: {torch_seed}")

def get_dataloaders(
    data_path: Path,
    percent_train_data: int,
    batch_size: int,
    shuffle_data_loader: bool,
    num_workers: int,
    output_path: Path,
) -> dict:

    # -- datasets --

    # Create datasets with different transforms
    train_transform = normalize_image_transform
    val_transform = normalize_image_transform
    # TODO: ^ Implement the augmentations, w/ each changing at every batch

    train_dataset = IAM_Dataset(
        root_dir=data_path,
        input_size=WordDetectorNet.input_size_ImageDimensions,
        output_size=WordDetectorNet.output_size_ImageDimensions,
        force_rebuild_cache=True,
        transform=train_transform,
    )
    val_dataset = IAM_Dataset(
        root_dir=data_path,
        input_size=WordDetectorNet.input_size_ImageDimensions,
        output_size=WordDetectorNet.output_size_ImageDimensions,
        force_rebuild_cache=True,
        transform=val_transform,
    )

    assert len(train_dataset) == len(val_dataset)

    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    split = int(percent_train_data / 100 * len(indices))

    train_indices = indices[:split]
    val_indices = indices[split:]

    with open(output_path / 'dataset_split_indices.json', 'w') as f:
        split_indices = {
            'train': train_indices,
            'val': val_indices,
        }
        json.dump(split_indices, f, indent=4)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_filenames = [sample['filename'] for sample in train_subset]
    val_filenames = [sample['filename'] for sample in val_subset]
    # Check that no train samples are in val
    assert len(set(train_filenames + val_filenames)) == len(train_filenames) + len(val_filenames)

    # -- dataloaders --

    dataloader_train = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_data_loader,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,  # or custom_collate_fn_with_padding
        pin_memory=True  # For faster GPU transfer
    )

    dataloader_val = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False, # no need to shuffle validation data and otherwise images break
        num_workers=num_workers,
        collate_fn=custom_collate_fn,  # or custom_collate_fn_with_padding
        pin_memory=True  # For faster GPU transfer
    )

    return {
        'train': dataloader_train,
        'val': dataloader_val,
    }

# TODO: Even w/o benchmarking, I can say that this is the bottleneck here. Also, it's
#       not the GPU part that is the bottleneck, but the CPU part. How about making
#       the loop annotated with (A) run in parallel?
def validate(net, dataloader_val, writer, device, input_size, output_size, regularisation=1e-8):
    global global_step
    net.eval()
    avg_loss = 0.0
    tp = 0
    fp = 0
    fn = 0
    image_counter = 0
    for i, batch in enumerate(dataloader_val):
        # For loss
        with torch.no_grad():
            images = batch['images']
            gt_encoded = batch['gt_encoded']
            images = images.to(device)
            gt_encoded = gt_encoded.to(device)
            y = net(images)
            loss = compute_loss(y, gt_encoded)
            avg_loss += loss.item()
        # For metrics; TODO: Can be combined by performing softmax on y from above but too lazy right now
        with torch.no_grad():
            images = batch['images']
            gt_encoded = batch['gt_encoded']
            images = images.to(device)
            gt_encoded = gt_encoded.to(device)
            y = net(images, apply_softmax=True)
            assert y[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND+1, :, :].min() >= 0.0
            assert y[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND+1, :, :].max() <= 1.0
        batch_size_here = y.shape[0]
        y = y.to('cpu').numpy()
        for i_element_in_batch in range(batch_size_here): # <-- (A)
            y_element = y[i_element_in_batch, :, :, :]
            decoded_aabbs = decode(y_element, scale=input_size.width / output_size.width, comp_fg=fg_by_cc(thres=0.5, max_num=1000))
            img_np = batch['images'][i_element_in_batch, 0, :, :].to('cpu').numpy()
            h, w = img_np.shape
            aabbs = [aabb.clip(BoundingBox(0, 0, w - 1, h - 1)) for aabb in decoded_aabbs]  # bounding box must be inside img
            clustered_aabbs = cluster_aabbs(aabbs)
            result = binary_classification_metrics(batch['bounding_boxes'][i_element_in_batch], clustered_aabbs)
            tp += result['tp']
            fp += result['fp']
            fn += result['fn']
            vis = draw_bboxes_on_image(img_np, clustered_aabbs)
            writer.add_image(f'img{image_counter}', vis.transpose((2, 0, 1)), global_step)
            image_counter += 1
    avg_loss = avg_loss / len(dataloader_val)
    precision = tp / (tp + fp + regularisation)
    recall = tp / (tp + fn + regularisation)
    f1 = 2*precision*recall / (precision + recall + regularisation)
    writer.add_scalar('loss/val', avg_loss, global_step)
    writer.add_scalar('f1/val', f1, global_step)
    return f1

def train(net, optimizer, loader, writer, device):
    global global_step

    net.train()
    for i, loader_item in enumerate(loader):

        images = loader_item['images']
        gt_encoded = loader_item['gt_encoded']

        images = images.to(device)
        gt_encoded = gt_encoded.to(device)

        # forward pass
        optimizer.zero_grad()
        y = net(images)
        loss = compute_loss(y, gt_encoded)

        # backward pass, optimize loss
        loss.backward()
        optimizer.step()

        # output
        print(f'{i + 1}/{len(loader)}: {loss.item()}')
        writer.add_scalar('loss/train', loss, global_step)
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
):
    writer = SummaryWriter(output_path / 'summary_writer')

    net = WordDetectorNet()
    net.to(device)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # main training loop
    epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    while True:
        epoch += 1
        print(f'Epoch: {epoch}')
        train(net, optimizer, dataloader_train, writer, device)
        if epoch % val_epoch == 0:
            f1 = validate(net, dataloader_val, writer, device, WordDetectorNet.input_size_ImageDimensions, WordDetectorNet.output_size_ImageDimensions)

            if f1 > best_val_f1:
                print(f"New best F1 score: {best_val_f1:.4f} -> {f1:.4f}, saving model.")
                best_val_f1 = f1
                torch.save(net.state_dict(), output_path / 'best_model.pth')
                patience_counter = 0
                with open(output_path / 'best_model.json', 'w') as f:
                    json.dump(
                        {
                            'epoch': epoch,
                            'global_step': global_step,
                            'f1': best_val_f1,
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
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            # TODO: Add seed in order to be able to run multiple seeds
        },
        {
            'best_val_f1': best_val_f1,
        }
    )

    writer.close()

    # TODO: Later, replace all print statements w/ proper logging statements.

    # TODO: Store result of training in a file, e.g., JSON or YAML.

def main(args: dict):

    args['output_path'].mkdir(exist_ok=True, parents=True)

    with open(args['output_path'] / 'args.json', 'w') as f:
        json.dump(
            args,
            f,
            indent=4,
            cls=CustomEncoder,
        )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_everything(
        numpy_seed=args['seed_split'],
        torch_seed=args['seed_model'],
        random_seed=args['seed_model'],
    )

    dataloaders = get_dataloaders(
        data_path=args['data_path'],
        percent_train_data=args['percent_train_data'],
        batch_size=args['batch_size'],
        shuffle_data_loader=args['shuffle_data_loader'],
        num_workers=args['num_workers'],
        output_path=args['output_path'],
    )

    train_network(
        output_path=args['output_path'],
        device=device,
        learning_rate=args['learning_rate'],
        batch_size=args['batch_size'],
        dataloader_train=dataloaders['train'],
        dataloader_val=dataloaders['val'],
        epoch_max=args['epoch_max'],
        patience_max=args['patience_max'],
        val_epoch=args['val_epoch'],
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)