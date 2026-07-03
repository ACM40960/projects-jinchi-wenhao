from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
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
class Box:
    """Bounding box in original-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def area(self) -> float:
        """Return box area in pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def grid_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Generate sliding-window positions and include the final image edge."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def tile_origins(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int]]:
    """Return top-left coordinates for every evaluation tile."""
    stride = max(1, int(round(tile_size * (1 - overlap))))
    xs = grid_positions(width, tile_size, stride)
    ys = grid_positions(height, tile_size, stride)
    return [(x, y) for y in ys for x in xs]


def iou(a: Box, b: Box) -> float:
    """Calculate intersection-over-union between two boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[Box], threshold: float) -> list[Box]:
    """Remove duplicate boxes by keeping the highest-confidence prediction first."""
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.conf, reverse=True):
        if all(iou(box, other) < threshold for other in kept):
            kept.append(box)
    return kept


def passes_filter(box: Box, preset: dict[str, float]) -> bool:
    """Apply an optional size/shape filter to reduce implausible detections."""
    width = max(0.0, box.x2 - box.x1)
    height = max(0.0, box.y2 - box.y1)
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


def read_labels(path: Path, width: int, height: int) -> list[Box]:
    """Read normalized YOLO labels and convert them to pixel-coordinate boxes."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    labels = []
    for line in text.splitlines():
        _class_id, xc, yc, bw, bh = line.split()
        xc_px = float(xc) * width
        yc_px = float(yc) * height
        bw_px = float(bw) * width
        bh_px = float(bh) * height
        labels.append(Box(xc_px - bw_px / 2, yc_px - bh_px / 2, xc_px + bw_px / 2, yc_px + bh_px / 2))
    return labels


def match_counts(predictions: list[Box], labels: list[Box], iou_threshold: float) -> tuple[int, int, int]:
    """
    Match predictions to ground-truth labels and count TP, FP, and FN.

    Each label can be matched at most once, using the highest-confidence
    predictions first.
    """
    matched: set[int] = set()
    tp = 0
    fp = 0
    for prediction in sorted(predictions, key=lambda item: item.conf, reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, label in enumerate(labels):
            if index in matched:
                continue
            score = iou(prediction, label)
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            matched.add(best_index)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(labels) - len(matched)


def predict_tiled(
    model: YOLO,
    image: Image.Image,
    tile_size: int,
    overlap: float,
    imgsz: int,
    conf: float,
    tile_nms_iou: float,
    final_nms_iou: float,
    filter_preset: str,
    max_detections: int,
    device: str,
) -> list[Box]:
    """
    Run tiled inference for evaluation and return merged predictions.

    This mirrors the final prediction pipeline but returns boxes for metric
    calculation instead of drawing images.
    """
    width, height = image.size
    predictions: list[Box] = []
    for x, y in tile_origins(width, height, tile_size, overlap):
        # Predict on each tile, then shift tile-local boxes back to original coordinates.
        tile = image.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
        result = model.predict(tile, imgsz=imgsz, conf=conf, iou=tile_nms_iou, device=device, verbose=False)[0]
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            predictions.append(Box(x1 + x, y1 + y, x2 + x, y2 + y, float(box.conf[0])))
    preset = FILTER_PRESETS[filter_preset]
    predictions = [box for box in predictions if passes_filter(box, preset)]
    kept = nms(predictions, final_nms_iou)
    if max_detections > 0:
        return kept[:max_detections]
    return kept


def main() -> None:
    """
    Evaluate tiled model predictions against original-image labels.

    Prints aggregate precision and recall for the selected threshold settings.
    """
    parser = argparse.ArgumentParser(description="Evaluate tiled YOLO inference on original images.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", default="Data/valid/images")
    parser.add_argument("--labels")
    parser.add_argument("--tile-size", type=int, default=416)
    parser.add_argument("--overlap", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--tile-nms-iou", type=float, default=0.7)
    parser.add_argument("--final-nms-iou", type=float, default=0.4)
    parser.add_argument("--filter-preset", default="none", choices=sorted(FILTER_PRESETS))
    parser.add_argument("--max-detections", type=int, default=0, help="Keep only the top-k detections per original image; 0 keeps all.")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-csv", help="Optional path for saving aggregate evaluation metrics.")
    parser.add_argument("--run-name", default="", help="Run name written into --save-csv.")
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels) if args.labels else image_dir.parent / "labels"
    model = YOLO(args.weights)
    total_tp = total_fp = total_fn = 0
    image_count = 0
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            labels = read_labels(label_dir / f"{image_path.stem}.txt", *image.size)
            predictions = predict_tiled(
                model=model,
                image=image,
                tile_size=args.tile_size,
                overlap=args.overlap,
                imgsz=args.imgsz,
                conf=args.conf,
                tile_nms_iou=args.tile_nms_iou,
                final_nms_iou=args.final_nms_iou,
                filter_preset=args.filter_preset,
                max_detections=args.max_detections,
                device=args.device,
            )
        tp, fp, fn = match_counts(predictions, labels, args.match_iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        image_count += 1

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    print(f"images={image_count} tp={total_tp} fp={total_fp} fn={total_fn}")
    print(f"precision={precision:.4f} recall={recall:.4f}")
    if args.save_csv:
        save_path = Path(args.save_csv)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "run",
                    "weights",
                    "images",
                    "tile_size",
                    "overlap",
                    "imgsz",
                    "conf",
                    "match_iou",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "run": args.run_name,
                    "weights": args.weights,
                    "images": str(image_dir),
                    "tile_size": args.tile_size,
                    "overlap": args.overlap,
                    "imgsz": args.imgsz,
                    "conf": args.conf,
                    "match_iou": args.match_iou,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                    "precision": f"{precision:.6f}",
                    "recall": f"{recall:.6f}",
                }
            )
        print(f"saved metrics -> {save_path}")


if __name__ == "__main__":
    main()
