"""Convert a YOLO segmentation dataset to a split COCO instance dataset.

The input must contain ``<split>/images`` and ``<split>/labels`` directories.
Each label row follows the YOLO segmentation layout:
``class_id x1 y1 x2 y2 ...`` with normalised polygon coordinates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.segmentation.cvat_coco import validate_coco_polygons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="YOLO dataset root")
    parser.add_argument("--output", type=Path, required=True, help="New COCO dataset root")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid"], help="Dataset splits to convert"
    )
    return parser.parse_args()


def jpeg_size(path: Path) -> tuple[int, int]:
    """Return a JPEG's width and height without an image-library dependency."""
    with path.open("rb") as source:
        if source.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG file: {path}")
        while True:
            marker_prefix = source.read(1)
            while marker_prefix == b"\xff":
                marker = source.read(1)
                if marker not in {b"\x00", b"\xff"}:
                    break
                marker_prefix = marker
            else:
                raise ValueError(f"Invalid JPEG marker in {path}")
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length = int.from_bytes(source.read(2), "big")
            if length < 2:
                raise ValueError(f"Invalid JPEG segment length in {path}")
            if marker in {
                b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7",
                b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf",
            }:
                payload = source.read(length - 2)
                return int.from_bytes(payload[3:5], "big"), int.from_bytes(payload[1:3], "big")
            source.seek(length - 2, 1)


def polygon_area(polygon: list[float]) -> float:
    points = list(zip(polygon[0::2], polygon[1::2]))
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def parse_label(path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    polygons: list[tuple[int, list[float]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 7 or len(values[1:]) % 2:
            raise ValueError(f"{path}:{line_number}: expected class ID plus >=3 coordinate pairs")
        class_id = int(values[0])
        coordinates = [float(value) for value in values[1:]]
        if not all(0 <= value <= 1 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: YOLO coordinates must be normalised to [0, 1]")
        polygon: list[float] = []
        for x, y in zip(coordinates[0::2], coordinates[1::2]):
            # COCO coordinates must lie strictly inside the image according to the
            # project's validator. Preserve boundary points just inside that edge.
            polygon.extend([min(x * width, width - 1e-6), min(y * height, height - 1e-6)])
        if polygon_area(polygon) <= 0:
            raise ValueError(f"{path}:{line_number}: polygon has zero area")
        polygons.append((class_id, polygon))
    return polygons


def convert_split(input_root: Path, output_root: Path, split: str) -> tuple[int, int]:
    source_images = input_root / split / "images"
    source_labels = input_root / split / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"Expected {source_images} and {source_labels}")

    destination_images = output_root / "images" / split
    destination_images.mkdir(parents=True, exist_ok=False)
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    for image_id, image_path in enumerate(sorted(source_images.glob("*.jpg")), start=1):
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
        width, height = jpeg_size(image_path)
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        for class_id, polygon in parse_label(label_path, width, height):
            xs, ys = polygon[0::2], polygon[1::2]
            x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "segmentation": [polygon],
                "area": polygon_area(polygon),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "iscrowd": 0,
            })
            annotation_id += 1
        shutil.copy2(image_path, destination_images / image_path.name)

    coco = {
        "info": {"description": f"100PA YOLO segmentation converted to COCO ({split})", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tooth", "supercategory": "tooth"}],
    }
    validate_coco_polygons(coco)
    annotation_path = output_root / "annotations" / f"instances_{split}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(images), len(annotations)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    totals = [convert_split(args.input, args.output, split) for split in args.splits]
    (args.output / "README.md").write_text(
        "# 100PA in COCO format\n\n"
        "- `images/<split>/`: original JPEG images\n"
        "- `annotations/instances_<split>.json`: COCO instance-segmentation annotations\n",
        encoding="utf-8",
    )
    print(f"Wrote {sum(images for images, _ in totals)} images and {sum(annotations for _, annotations in totals)} annotations to {args.output}")


if __name__ == "__main__":
    main()
