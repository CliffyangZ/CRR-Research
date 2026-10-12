import torch

from research.models.unet import UNet
from src.segmentation.losses import bce_dice_loss


def test_forward_backward_and_reload(tmp_path):
    torch.manual_seed(0)
    net = UNet(base_channels=8, depth=3)
    x = torch.rand(2, 1, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.5).float()
    out = net(x)
    assert out.shape == (2, 1, 64, 64)
    loss = bce_dice_loss(out, y)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in net.parameters())
    torch.save(net.state_dict(), tmp_path / "w.pt")
    net2 = UNet(base_channels=8, depth=3)
    net2.load_state_dict(torch.load(tmp_path / "w.pt"))
    net.eval(); net2.eval()
    assert torch.allclose(net(x), net2(x), atol=1e-6)


def test_baseline_size_512_shape():
    from src.segmentation.config import Model
    from research.models.unet import build_unet
    net = build_unet(Model()).eval()
    assert net(torch.zeros(1, 1, 512, 512)).shape == (1, 1, 512, 512)
    assert 7e6 < sum(p.numel() for p in net.parameters()) < 9e6
