"""Fast R-CNN network (VGG16 backbone), after rbgirshick/fast-rcnn models/VGG16/train.prototxt."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16
from torchvision.ops import RoIPool

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FastRCNN(nn.Module):
    def __init__(self, num_fg_classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.k = num_fg_classes
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = vgg.features[:-1]  # drop pool5 -> conv5_3 at stride 16
        for p in self.features[:10].parameters():  # conv1_x, conv2_x frozen (as in the reference)
            p.requires_grad = False
        self.roi_pool = RoIPool((7, 7), spatial_scale=1 / 16)
        c = vgg.classifier
        self.fc = nn.Sequential(c[0], c[1], c[2], c[3], c[4], c[5])  # fc6, relu, drop, fc7, relu, drop
        self.cls_score = nn.Linear(4096, num_fg_classes + 1)
        self.bbox_pred = nn.Linear(4096, 4 * num_fg_classes)
        nn.init.normal_(self.cls_score.weight, std=0.01); nn.init.zeros_(self.cls_score.bias)
        nn.init.normal_(self.bbox_pred.weight, std=0.001); nn.init.zeros_(self.bbox_pred.bias)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def forward(self, images: torch.Tensor, rois: list[torch.Tensor]):
        """images: [B,3,H,W] float in [0,1] RGB; rois: B tensors [Ni,4] xyxy pixels."""
        x = (images - self.mean) / self.std
        feat = self.features(x)
        pooled = self.roi_pool(feat, rois)
        h = self.fc(pooled.flatten(1))
        return self.cls_score(h), self.bbox_pred(h)


def fast_rcnn_loss(cls_logits, deltas, labels, targets, lam: float = 1.0):
    """L = CE(p, u) + lam * [u >= 1] * smooth_L1(t_u - v)   (paper eq. 1; one fg class -> t_u = deltas)."""
    l_cls = F.cross_entropy(cls_logits, labels)
    fg = labels > 0
    l_box = F.smooth_l1_loss(deltas[fg], targets[fg], beta=1.0, reduction="sum") / max(1, labels.numel()) if fg.any() else deltas.sum() * 0
    return l_cls + lam * l_box, l_cls.detach(), l_box.detach()
