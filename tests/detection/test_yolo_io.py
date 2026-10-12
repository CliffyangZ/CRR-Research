import numpy as np
import pytest

from src.detection.yolo_io import clip_boxes, read_yolo_labels, validate_boxes, write_yolo_labels


def test_roundtrip(tmp_path):
    b = np.array([[10.0, 20.0, 110.0, 220.0]])
    write_yolo_labels(tmp_path / "a.txt", np.array([0]), b, 400, 300)
    c, r = read_yolo_labels(tmp_path / "a.txt", 400, 300)
    assert c.tolist() == [0] and np.allclose(r, b, atol=0.01)


def test_missing_label_is_error_not_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_yolo_labels(tmp_path / "none.txt", 10, 10)


def test_validation():
    assert validate_boxes(np.array([[0, 0, 10, 10]]), 10, 10) == []
    assert validate_boxes(np.array([[-0.0012, 0, 10, 10]]), 10, 10) == []  # rounding only
    assert validate_boxes(np.array([[-5, 0, 10, 10]]), 10, 10)
    assert validate_boxes(np.array([[3, 3, 3, 9]]), 10, 10)
    assert clip_boxes(np.array([[-0.0012, 0, 10, 10]]), 10, 10)[0, 0] == 0
