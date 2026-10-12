import cv2
import numpy as np

from src.detection.inventory import inspect_image, mark_duplicates


def _w(p, a):
    cv2.imwrite(str(p), a)
    return p


def test_ok_gray_and_rgb_gray(tmp_path):
    g = (np.random.RandomState(0).rand(64, 48) * 255).astype(np.uint8)
    a = inspect_image(_w(tmp_path / "a.png", g))
    b = inspect_image(_w(tmp_path / "b.png", np.dstack([g] * 3)))
    assert a.status == b.status == "included"
    assert a.channels == 1 and b.channels == 3 and b.channel_mean_dev == 0
    assert a.bit_depth == 8 and a.phash


def test_corrupt(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image")
    it = inspect_image(p)
    assert it.status == "excluded" and it.reason.startswith("corrupt")


def test_non_8bit_quarantined(tmp_path):
    a16 = (np.random.RandomState(1).rand(32, 32) * 65535).astype(np.uint16)
    it = inspect_image(_w(tmp_path / "x.png", a16))
    assert it.status == "quarantined" and it.reason == "non_8bit" and it.bit_depth == 16


def test_color_quarantined(tmp_path):
    c = np.zeros((32, 32, 3), np.uint8)
    c[..., 2] = 200
    it = inspect_image(_w(tmp_path / "c.png", c))
    assert it.status == "quarantined" and it.reason == "color_content"


def test_duplicates_excluded_with_reason(tmp_path):
    g = (np.random.RandomState(2).rand(32, 32) * 255).astype(np.uint8)
    items = [inspect_image(_w(tmp_path / f"{n}.png", g)) for n in ("1", "2")]
    mark_duplicates(items)
    assert items[0].status == "included"
    assert items[1].status == "excluded" and items[1].reason == "duplicate_of:1"
