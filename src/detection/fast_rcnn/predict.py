"""Fast R-CNN inference: score every proposal, decode class-specific boxes, NMS (0.3 as in the paper)."""

from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import nms

from .boxes import clip, decode


@torch.no_grad()
def detect(model, image: torch.Tensor, proposals: torch.Tensor, device, score_min=0.001, nms_iou=0.3, max_det=300):
    """image [3,H,W] -> (boxes[N,4] xyxy, scores[N]) in image pixels."""
    model.eval()
    h, w = image.shape[1:]
    if len(proposals) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    cls, deltas = model(image[None].to(device), [proposals.to(device)])
    scores = torch.softmax(cls.float(), 1)[:, 1]
    boxes = clip(decode(proposals.to(device), deltas.float()[:, :4]), h, w)
    keep = (scores >= score_min) & ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
    boxes, scores = boxes[keep], scores[keep]
    k = nms(boxes, scores, nms_iou)[:max_det]
    return boxes[k].cpu().numpy(), scores[k].cpu().numpy()


def load_fast_rcnn(weights_path: str, device):
    from .model import FastRCNN

    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    m = FastRCNN(ck["num_fg_classes"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
