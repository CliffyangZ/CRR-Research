"""Align YOLO-labelled tooth crops to processed images and export COCO JSON.

100PA names such as ``102_01`` and ``102_02`` are distinct tooth crops from
``data/processed/102.png``.  Their segmentation coordinates must therefore be
translated back to the matching crop location before importing into a CVAT task
containing the processed images.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.segmentation.cvat_coco import validate_coco_polygons
from yolo_segmentation_to_coco import parse_label, polygon_area


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="100PA YOLO dataset root")
    parser.add_argument("--processed", type=Path, required=True, help="Full processed PNG image directory")
    parser.add_argument("--output", type=Path, required=True, help="COCO JSON to write")
    parser.add_argument(
        "--min-score", type=float, default=0.20, help="Fail when crop-match correlation is below this value"
    )
    return parser.parse_args()


def grayscale(path: Path) -> np.ndarray:
    """Decode an image through ffmpeg, avoiding an additional Python image dependency."""
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)],
            text=True,
        )
    )["streams"][0]
    width, height = int(probe["width"]), int(probe["height"])
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width).astype(np.float64)


def match_crop(crop: np.ndarray, image: np.ndarray) -> tuple[int, int, float]:
    """Find crop's top-left position using FFT normalised cross-correlation."""
    crop_height, crop_width = crop.shape
    image_height, image_width = image.shape
    if crop_height > image_height or crop_width > image_width:
        raise ValueError("Crop is larger than its corresponding processed image")
    centred = crop - crop.mean()
    shape = (image_height + crop_height - 1, image_width + crop_width - 1)
    correlation = np.fft.irfft2(
        np.fft.rfft2(image, shape) * np.fft.rfft2(centred[::-1, ::-1], shape), shape
    )[crop_height - 1 : image_height, crop_width - 1 : image_width]
    squared = np.pad(np.cumsum(np.cumsum(image * image, axis=0), axis=1), ((1, 0), (1, 0)))
    window_energy = (
        squared[crop_height:, crop_width:]
        - squared[:-crop_height, crop_width:]
        - squared[crop_height:, :-crop_width]
        + squared[:-crop_height, :-crop_width]
    )
    score = correlation / np.sqrt(window_energy * np.sum(centred * centred))
    top, left = np.unravel_index(np.argmax(score), score.shape)
    return int(left), int(top), float(score[top, left])


def processed_stem(crop_path: Path) -> str:
    # ``102_01_jpg.rf.<hash>.jpg`` becomes ``102``.
    return crop_path.name.split("_", 1)[0]


def main() -> None:
    args = parse_args()
    crops = sorted(path for split in ("train", "valid") for path in (args.input / split / "images").glob("*.jpg"))
    grouped: dict[str, list[Path]] = defaultdict(list)
    for crop in crops:
        grouped[processed_stem(crop)].append(crop)

    images: list[dict] = []
    annotations: list[dict] = []
    alignments: list[dict] = []
    missing_processed: list[dict] = []
    annotation_id = 1
    for image_id, stem in enumerate(sorted(grouped, key=int), start=1):
        processed_path = args.processed / f"{stem}.png"
        if not processed_path.is_file():
            missing_processed.append(
                {"processed_file": processed_path.name, "crops": [crop.name for crop in grouped[stem]]}
            )
            continue
        processed = grayscale(processed_path)
        height, width = processed.shape
        images.append({"id": image_id, "file_name": processed_path.name, "width": width, "height": height})
        for crop_path in grouped[stem]:
            crop = grayscale(crop_path)
            left, top, score = match_crop(crop, processed)
            if score < args.min_score:
                raise ValueError(f"Unreliable match for {crop_path.name}: {score:.3f} < {args.min_score:.3f}")
            label_path = crop_path.parent.parent / "labels" / f"{crop_path.stem}.txt"
            for class_id, polygon in parse_label(label_path, crop.shape[1], crop.shape[0]):
                projected = [coordinate + (left if index % 2 == 0 else top) for index, coordinate in enumerate(polygon)]
                xs, ys = projected[0::2], projected[1::2]
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "segmentation": [projected],
                    "area": polygon_area(projected),
                    "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "iscrowd": 0,
                })
                annotation_id += 1
            alignments.append({"crop": crop_path.name, "processed_file": processed_path.name, "x": left, "y": top, "score": round(score, 6)})

    coco = {
        "info": {"description": "100PA crop annotations aligned to processed images for CVAT", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tooth", "supercategory": "tooth"}],
    }
    validate_coco_polygons(coco)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_name(args.output.stem + "_alignment_report.json").write_text(
        json.dumps(alignments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_name(args.output.stem + "_missing_processed.json").write_text(
        json.dumps(missing_processed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(images)} processed images and {len(annotations)} annotations to {args.output}")
    print(f"Lowest alignment score: {min(item['score'] for item in alignments):.3f}")
    if missing_processed:
        print(f"Skipped {len(missing_processed)} absent processed images; see *_missing_processed.json")


if __name__ == "__main__":
    main()
