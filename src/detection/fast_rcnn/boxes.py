"""Box coding and RoI sampling for Fast R-CNN (xyxy pixel boxes, torch tensors)."""

from __future__ import annotations

import math

import torch
from torchvision.ops import box_iou

BBOX_STDS = (0.1, 0.1, 0.2, 0.2)  # same normalization as fast-rcnn BBOX_NORMALIZE_STDS (means = 0)
_CLIP = math.log(1000.0 / 16)


def encode(rois: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """(dx, dy, dw, dh) targets of gt w.r.t. rois, divided by BBOX_STDS."""
    w, h = rois[:, 2] - rois[:, 0], rois[:, 3] - rois[:, 1]
    cx, cy = rois[:, 0] + 0.5 * w, rois[:, 1] + 0.5 * h
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    t = torch.stack([(gx - cx) / w, (gy - cy) / h, torch.log(gw / w), torch.log(gh / h)], 1)
    return t / t.new_tensor(BBOX_STDS)


def decode(rois: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    d = deltas * deltas.new_tensor(BBOX_STDS)
    w, h = rois[:, 2] - rois[:, 0], rois[:, 3] - rois[:, 1]
    cx, cy = rois[:, 0] + 0.5 * w, rois[:, 1] + 0.5 * h
    px, py = d[:, 0] * w + cx, d[:, 1] * h + cy
    pw, ph = torch.exp(d[:, 2].clamp(max=_CLIP)) * w, torch.exp(d[:, 3].clamp(max=_CLIP)) * h
    return torch.stack([px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph], 1)


def clip(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b = boxes.clone()
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
    return b


def sample_rois(
    proposals: torch.Tensor, gt: torch.Tensor, rois_per_image: int = 64, fg_fraction: float = 0.25,
    fg_thresh: float = 0.5, bg_hi: float = 0.5, bg_lo: float = 0.1, gen: torch.Generator | None = None,
):
    """Fast R-CNN mini-batch sampling for ONE image (single foreground class).

    GT boxes are appended to the proposals (as in the reference implementation).
    Returns rois[N,4], labels[N] (1 = fg, 0 = bg), bbox_targets[N,4] (zeros for bg).
    """
    rois = torch.cat([proposals, gt], 0)
    iou = box_iou(rois, gt)
    max_iou, arg = iou.max(1)
    fg = torch.where(max_iou >= fg_thresh)[0]
    bg = torch.where((max_iou < bg_hi) & (max_iou >= bg_lo))[0]
    n_fg = min(int(round(rois_per_image * fg_fraction)), len(fg))
    n_bg = min(rois_per_image - n_fg, len(bg))
    fg = fg[torch.randperm(len(fg), generator=gen)[:n_fg]]
    bg = bg[torch.randperm(len(bg), generator=gen)[:n_bg]]
    keep = torch.cat([fg, bg])
    labels = torch.zeros(len(keep), dtype=torch.long)
    labels[: len(fg)] = 1
    sel = rois[keep]
    targets = torch.zeros(len(keep), 4)
    if len(fg):
        targets[: len(fg)] = encode(sel[: len(fg)], gt[arg[fg]])
    return sel, labels, targets
