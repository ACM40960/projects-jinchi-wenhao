from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILTER_PRESETS = {
    "none": {},
    "loose_person": {"min_ratio": 0.45, "max_ratio": 4.0, "min_short": 2.0, "max_long": 260.0},
    "person": {"min_ratio": 0.60, "max_ratio": 3.4, "min_short": 4.0, "max_long": 220.0},
    "tight_person": {"min_ratio": 0.70, "max_ratio": 3.0, "min_short": 5.0, "max_long": 180.0},
    "no_large": {"max_short": 140.0, "max_long": 260.0},
}


@dataclass(frozen=True)
class Detection:
    """Final detection box in original-image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def area(self) -> float:
        """Return detection area in pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def grid_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """
    Generate tile start positions for one image dimension.

    The last tile is aligned to the image edge so the full image is covered.
    """
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def tile_origins(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int]]:
    """Return top-left coordinates for all sliding-window tiles."""
    stride = max(1, int(round(tile_size * (1 - overlap))))
    return [(x, y) for y in grid_positions(height, tile_size, stride) for x in grid_positions(width, tile_size, stride)]


def iou(a: Detection, b: Detection) -> float:
    """Calculate intersection-over-union between two detection boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """
    Apply non-maximum suppression across all tiles.

    This removes duplicate detections produced where neighboring tiles overlap.
    """
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.conf, reverse=True):
        if all(iou(detection, existing) < iou_threshold for existing in kept):
            kept.append(detection)
    return kept


def passes_filter(detection: Detection, preset: dict[str, float]) -> bool:
    """
    Check whether a detection matches an optional shape-size filter.

    Filters are used only when we want to remove boxes that are clearly too
    large, too small, or unlikely to be person-shaped.
    """
    width = max(0.0, detection.x2 - detection.x1)
    height = max(0.0, detection.y2 - detection.y1)
    if width <= 0 or height <= 0:
        return False
    ratio = height / width
    short = min(width, height)
    long = max(width, height)
    area = width * height
    checks = {
        "min_ratio": ratio,
        "max_ratio": ratio,
        "min_short": short,
        "max_short": short,
        "min_long": long,
        "max_long": long,
        "min_area": area,
        "max_area": area,
    }
    for key, value in preset.items():
        if key.startswith("min_") and checks[key] < value:
            return False
        if key.startswith("max_") and checks[key] > value:
            return False
    return True


def source_images(source: Path) -> list[Path]:
    """Return one image path or all supported images inside a folder."""
    if source.is_file():
        return [source]
    return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """
    Draw predicted boxes and zoomed insets on a copy of the original image.

    The inset makes small Waldo detections visible in saved prediction images.
    """
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, detection in enumerate(detections, start=1):
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        center_x = (detection.x1 + detection.x2) / 2
        center_y = (detection.y1 + detection.y2) / 2
        marker = max(12, int(max(detection.x2 - detection.x1, detection.y2 - detection.y1) * 0.45))

        draw.rectangle(box, outline="red", width=4)

        # Draw the confidence label directly above the detection box.
        label = f"waldo {detection.conf:.2f}"
        text_x = max(0, detection.x1)
        text_y = max(0, detection.y1 - 14)
        text_box = draw.textbbox((text_x, text_y), label, font=font)
        padded_box = (text_box[0] - 4, text_box[1] - 3, text_box[2] + 4, text_box[3] + 3)
        draw.rectangle(padded_box, fill="red", outline="white", width=2)
        draw.text((text_x, text_y), label, fill="white", font=font)

        crop_pad = max(80, int(marker * 3))
        crop_box = (
            max(0, int(center_x - crop_pad)),
            max(0, int(center_y - crop_pad)),
            min(canvas.width, int(center_x + crop_pad)),
            min(canvas.height, int(center_y + crop_pad)),
        )
        crop = image.crop(crop_box)
        # Enlarge the local crop so the tiny target can be inspected easily.
        zoom_width = min(360, max(180, crop.width * 3))
        zoom_height = max(1, int(crop.height * (zoom_width / max(1, crop.width))))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        crop = crop.resize((zoom_width, zoom_height), resample)

        candidates = [
            (16, 16 + (index - 1) * (zoom_height + 22)),
            (canvas.width - zoom_width - 16, 16 + (index - 1) * (zoom_height + 22)),
            (16, canvas.height - zoom_height - 16 - (index - 1) * (zoom_height + 22)),
            (canvas.width - zoom_width - 16, canvas.height - zoom_height - 16 - (index - 1) * (zoom_height + 22)),
        ]
        inset_x, inset_y = choose_inset_position(candidates, zoom_width, zoom_height, detection, canvas.width, canvas.height)
        # Place the zoom window where it overlaps the detection as little as possible.
        canvas.paste(crop, (inset_x, inset_y))
        inset_box = (inset_x, inset_y, inset_x + zoom_width, inset_y + zoom_height)
        draw.rectangle(inset_box, outline="red", width=5)

        inset_center = (inset_x + zoom_width / 2, inset_y + zoom_height / 2)
        draw.line((inset_box[2], inset_box[3], center_x, center_y), fill="yellow", width=4)
        draw.line((inset_box[2], inset_box[3], center_x, center_y), fill="black", width=1)

        zoom_label = f"zoom #{index}"
        zoom_text_box = draw.textbbox((inset_x + 8, inset_y + 8), zoom_label, font=font)
        zoom_padded = (zoom_text_box[0] - 4, zoom_text_box[1] - 3, zoom_text_box[2] + 4, zoom_text_box[3] + 3)
        draw.rectangle(zoom_padded, fill="red", outline="white", width=2)
        draw.text((inset_x + 8, inset_y + 8), zoom_label, fill="white", font=font)
    return canvas


def choose_inset_position(
    candidates: list[tuple[int, int]],
    width: int,
    height: int,
    detection: Detection,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """
    Pick an inset location that avoids covering the detected Waldo box.

    Candidate corners are scored by overlap with the detection box.
    """
    valid_candidates = []
    for x, y in candidates:
        x = min(max(16, x), max(16, image_width - width - 16))
        y = min(max(16, y), max(16, image_height - height - 16))
        inset = Detection(float(x), float(y), float(x + width), float(y + height), 1.0)
        valid_candidates.append((iou(inset, detection), x, y))
    _overlap, best_x, best_y = min(valid_candidates, key=lambda item: item[0])
    return best_x, best_y


def save_label_file(label_path: Path, image_size: tuple[int, int], detections: list[Detection]) -> None:
    """
    Save final detections as YOLO-style labels with confidence appended.

    Coordinates are normalized relative to the original image size.
    """
    width, height = image_size
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for detection in detections:
        box_width = max(0.0, detection.x2 - detection.x1)
        box_height = max(0.0, detection.y2 - detection.y1)
        x_center = detection.x1 + box_width / 2
        y_center = detection.y1 + box_height / 2
        lines.append(
            "0 "
            f"{x_center / width:.6f} "
            f"{y_center / height:.6f} "
            f"{box_width / width:.6f} "
            f"{box_height / height:.6f} "
            f"{detection.conf:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def predict_image(
    model: YOLO,
    image_path: Path,
    tile_size: int,
    overlap: float,
    conf: float,
    model_iou: float,
    final_iou: float,
    imgsz: int,
    filter_preset: str,
    max_detections: int,
) -> list[Detection]:
    """
    Predict Waldo on one large image by running YOLO over overlapping tiles.

    Tile-level boxes are shifted back into original-image coordinates, filtered,
    and merged with final NMS.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        detections: list[Detection] = []

        for x, y in tile_origins(width, height, tile_size, overlap):
            # Crop the tile, run YOLO, then translate tile boxes back to full-image coordinates.
            tile = image.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
            results = model.predict(tile, imgsz=imgsz, conf=conf, iou=model_iou, verbose=False)
            boxes = results[0].boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        x1=x1 + x,
                        y1=y1 + y,
                        x2=x2 + x,
                        y2=y2 + y,
                        conf=float(box.conf[0]),
                    )
                )

        preset = FILTER_PRESETS[filter_preset]
        # Shape filtering happens before cross-tile NMS to reduce obvious false positives.
        detections = [detection for detection in detections if passes_filter(detection, preset)]
        kept = nms(detections, final_iou)
        if max_detections > 0:
            return kept[:max_detections]
        return kept


def main() -> None:
    """
    Command-line entry point for tiled prediction on original images.

    Outputs annotated images and label files under the selected output folder.
    """
    parser = argparse.ArgumentParser(description="Predict Waldo on large images using tiled YOLO inference.")
    parser.add_argument("--weights", required=True, help="Trained tiled YOLO weights.")
    parser.add_argument("--source", required=True, help="Image or folder to predict.")
    parser.add_argument("--tile-size", type=int, default=768, help="Tile size used at inference.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio.")
    parser.add_argument("--imgsz", type=int, default=768, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO tile-level NMS IoU.")
    parser.add_argument("--final-iou", type=float, default=0.4, help="Final cross-tile NMS IoU.")
    parser.add_argument("--filter-preset", default="none", choices=sorted(FILTER_PRESETS), help="Optional shape filter for detections.")
    parser.add_argument("--max-detections", type=int, default=0, help="Keep only the top-k detections per image; 0 keeps all.")
    parser.add_argument("--out", default="runs/predict_tiled/waldo", help="Output folder.")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    for image_path in source_images(Path(args.source)):
        detections = predict_image(
            model=model,
            image_path=image_path,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            model_iou=args.iou,
            final_iou=args.final_iou,
            imgsz=args.imgsz,
            filter_preset=args.filter_preset,
            max_detections=args.max_detections,
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            drawn = draw_detections(image, detections)
            out_path = out_dir / image_path.name
            drawn.save(out_path, quality=95)
            save_label_file(out_dir / "labels" / f"{image_path.stem}.txt", image.size, detections)

        print(f"{image_path}: {len(detections)} detections -> {out_path}")
        print(f"  labels -> {out_dir / 'labels' / f'{image_path.stem}.txt'}")
        for index, detection in enumerate(detections, start=1):
            print(
                f"  #{index} conf={detection.conf:.3f} "
                f"xyxy=({detection.x1:.1f}, {detection.y1:.1f}, "
                f"{detection.x2:.1f}, {detection.y2:.1f})"
            )


if __name__ == "__main__":
    main()
