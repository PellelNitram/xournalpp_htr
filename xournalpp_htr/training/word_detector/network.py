"""WordDetectorNN network architecture and training loss.

Originally created by `Harald Scheidl <https://github.com/githubharald/WordDetectorNN>`_;
reimplemented here with some best practices for integration into Xournal++ HTR.
Requires the ``training-word-detector`` extra (torch/torchvision).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet

from xournalpp_htr.training.shared.bounding_box import ImageDimensions
from xournalpp_htr.training.shared.postprocessing import MapOrdering
from xournalpp_htr.training.word_detector.config import ModelConfig


class ModifiedResNet18(ResNet):
    """ResNet-18 backbone returning intermediate U-Net-style features.

    The first conv is adapted for 1-channel (grayscale) input and the final fc
    layer is removed; ``forward`` returns ``(bb5, bb4, bb3, bb2, bb1)``.
    """

    def __init__(self, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=1000, **kwargs)

        # Grayscale (1-channel) input instead of the original 3-channel.
        original_conv1 = self.conv1
        self.conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        # The fully connected head is unused for feature extraction.
        del self.fc

    def _forward_impl(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu(x)  # bb1 (before maxpool)
        x = self.maxpool(out1)

        out2 = self.layer1(x)  # bb2
        out3 = self.layer2(out2)  # bb3
        out4 = self.layer3(out3)  # bb4
        out5 = self.layer4(out4)  # bb5

        return out5, out4, out3, out2, out1

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)


class UpscaleAndConcatLayer(torch.nn.Module):
    """Upscale a small map, concat with a larger map, conv to ``cz`` channels."""

    def __init__(self, cx, cy, cz):
        super().__init__()
        self.conv = torch.nn.Conv2d(cx + cy, cz, 3, padding=1)

    def forward(self, x, y, s):
        x = F.interpolate(x, s)
        z = torch.cat((x, y), 1)
        return F.relu(self.conv(z))


class WordDetectorNet(torch.nn.Module):
    _defaults = ModelConfig()
    input_size = (_defaults.input_height, _defaults.input_width)
    output_size = (_defaults.output_height, _defaults.output_width)
    input_size_ImageDimensions: ImageDimensions = ImageDimensions(
        width=_defaults.input_width, height=_defaults.input_height
    )
    output_size_ImageDimensions: ImageDimensions = ImageDimensions(
        width=_defaults.output_width, height=_defaults.output_height
    )

    def __init__(self):
        super().__init__()

        self.backbone = ModifiedResNet18()  # randomly initialised weights

        self.up1 = UpscaleAndConcatLayer(512, 256, 256)  # input//16
        self.up2 = UpscaleAndConcatLayer(256, 128, 128)  # input//8
        self.up3 = UpscaleAndConcatLayer(128, 64, 64)  # input//4
        self.up4 = UpscaleAndConcatLayer(64, 64, 32)  # input//2

        self.conv1 = torch.nn.Conv2d(32, MapOrdering.NUM_MAPS, 3, 1, padding=1)

    @staticmethod
    def scale_shape(s, f):
        assert s[0] % f == 0 and s[1] % f == 0
        return s[0] // f, s[1] // f

    def output_activation(self, x, apply_softmax):
        if apply_softmax:
            seg = torch.softmax(
                x[:, MapOrdering.SEG_WORD : MapOrdering.SEG_BACKGROUND + 1], dim=1
            )
        else:
            seg = x[:, MapOrdering.SEG_WORD : MapOrdering.SEG_BACKGROUND + 1]
        geo = torch.sigmoid(x[:, MapOrdering.GEO_TOP :]) * self.input_size[0]
        return torch.cat([seg, geo], dim=1)

    def forward(self, x, apply_softmax=False):
        s = x.shape[2:]  # Original image shape HxW
        bb5, bb4, bb3, bb2, bb1 = self.backbone(x)

        y = self.up1(bb5, bb4, self.scale_shape(s, 16))
        y = self.up2(y, bb3, self.scale_shape(s, 8))
        y = self.up3(y, bb2, self.scale_shape(s, 4))
        y = self.up4(y, bb1, self.scale_shape(s, 2))

        y = self.conv1(y)
        return self.output_activation(y, apply_softmax)


def compute_loss(y, gt_map):
    # 1. segmentation loss
    target_labels = torch.argmax(
        gt_map[:, MapOrdering.SEG_WORD : MapOrdering.SEG_BACKGROUND + 1], dim=1
    )
    loss_seg = F.cross_entropy(
        y[:, MapOrdering.SEG_WORD : MapOrdering.SEG_BACKGROUND + 1], target_labels
    )

    # 2. geometry loss -- distances to all sides of the aabb
    t = torch.minimum(y[:, MapOrdering.GEO_TOP], gt_map[:, MapOrdering.GEO_TOP])
    b = torch.minimum(y[:, MapOrdering.GEO_BOTTOM], gt_map[:, MapOrdering.GEO_BOTTOM])
    l = torch.minimum(y[:, MapOrdering.GEO_LEFT], gt_map[:, MapOrdering.GEO_LEFT])  # noqa: E741
    r = torch.minimum(y[:, MapOrdering.GEO_RIGHT], gt_map[:, MapOrdering.GEO_RIGHT])

    y_width = y[:, MapOrdering.GEO_LEFT, ...] + y[:, MapOrdering.GEO_RIGHT, ...]
    y_height = y[:, MapOrdering.GEO_TOP, ...] + y[:, MapOrdering.GEO_BOTTOM, ...]
    area1 = y_width * y_height

    gt_width = (
        gt_map[:, MapOrdering.GEO_LEFT, ...] + gt_map[:, MapOrdering.GEO_RIGHT, ...]
    )
    gt_height = (
        gt_map[:, MapOrdering.GEO_TOP, ...] + gt_map[:, MapOrdering.GEO_BOTTOM, ...]
    )
    area2 = gt_width * gt_height

    intersection = (r + l) * (b + t)
    union = area1 + area2 - intersection
    eps = 0.01  # avoid division by 0
    iou = intersection / (union + eps)
    iou = iou[gt_map[:, MapOrdering.SEG_WORD] > 0]
    loss_aabb = -torch.log(torch.mean(iou))

    return loss_seg + loss_aabb
