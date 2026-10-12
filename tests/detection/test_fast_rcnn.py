import numpy as np
import pytest
import torch

from src.detection.fast_rcnn.boxes import clip, decode, encode, sample_rois
from src.detection.fast_rcnn.model import FastRCNN, fast_rcnn_loss
from src.detection.fast_rcnn.predict import detect


def test_encode_decode_roundtrip():
    rois = torch.tensor([[10., 20., 110., 220.], [0., 0., 50., 80.]])
    gt = torch.tensor([[15., 30., 120., 200.], [5., 5., 60., 90.]])
    assert torch.allclose(decode(rois, encode(rois, gt)), gt, atol=1e-3)
    assert torch.allclose(encode(rois, rois), torch.zeros(2, 4), atol=1e-6)


def test_sampling_fractions_and_thresholds():
    g = torch.Generator().manual_seed(0)
    gt = torch.tensor([[100., 100., 200., 300.]])
    rng = np.random.default_rng(0)
    jit = torch.tensor(rng.uniform(-40, 40, (300, 4)), dtype=torch.float32)
    props = gt.repeat(300, 1) + jit
    rois, labels, targets = sample_rois(props, gt, 64, 0.25, gen=g)
    assert len(rois) <= 64 and (labels == 1).sum() <= 16
    from torchvision.ops import box_iou
    iou = box_iou(rois, gt).squeeze(1)
    assert (iou[labels == 1] >= 0.5).all() and (iou[labels == 0] < 0.5).all() and (iou[labels == 0] >= 0.1).all()
    assert (targets[labels == 0] == 0).all()
    # GT itself is always a foreground candidate (appended to proposals)
    r2, l2, _ = sample_rois(torch.zeros(0, 4), gt, 64, 0.25, gen=g)
    assert l2.tolist() == [1]


def test_model_shapes_and_loss_backward():
    m = FastRCNN(1, pretrained=False)
    assert all(not p.requires_grad for p in m.features[:10].parameters())  # conv1/conv2 frozen
    imgs = torch.rand(2, 3, 128, 128)
    rois = [torch.tensor([[0., 0., 64., 64.], [10., 10., 120., 100.]]), torch.tensor([[5., 5., 100., 120.]])]
    cls, dl = m(imgs, rois)
    assert cls.shape == (3, 2) and dl.shape == (3, 4)
    loss, lc, lb = fast_rcnn_loss(cls, dl, torch.tensor([1, 0, 0]), torch.zeros(3, 4))
    loss.backward()
    assert m.cls_score.weight.grad is not None and m.features[10].weight.grad is not None


def test_detect_outputs_in_image_and_nms():
    m = FastRCNN(1, pretrained=False)
    img = torch.rand(3, 128, 128)
    props = torch.tensor([[0., 0., 100., 100.], [2., 2., 102., 102.], [30., 30., 120., 120.]])
    b, s = detect(m, img, props, "cpu", score_min=0.0)
    assert b.shape[1] == 4 and len(b) == len(s) <= 3
    assert (b >= 0).all() and (b <= 128).all()
    assert detect(m, img, torch.zeros(0, 4), "cpu")[0].shape == (0, 4)
