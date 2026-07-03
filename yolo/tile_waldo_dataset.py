from __future__ import annotations #

import argparse
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    """Pixel-coordinate bounding box read from a YOLO label file."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        """Return the visible box area in pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Tile:
    """Image crop window described by top-left and bottom-right coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Return tile width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Return tile height in pixels."""
        return self.y2 - self.y1


def load_yaml(path: Path) -> dict:
    """
    Load a YOLO data.yaml file and return it as a dictionary.

    Raises an error if the YAML content is not a mapping.
    """
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml: {path}")
    return data


def split_path(data_yaml: Path, value: str) -> Path:
    """Resolve a split path from data.yaml relative to the YAML file location."""
    path = Path(value)
    return path if path.is_absolute() else data_yaml.parent / path


def image_files(image_dir: Path) -> list[Path]:
    """Return supported image files in a deterministic order."""
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[Box]:
    """
    Convert normalized YOLO labels into pixel-coordinate boxes.

    Empty or missing label files are treated as background images.
    """
    if not label_path.exists():
        return []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    boxes: list[Box] = []
    for line in text.splitlines():
        class_id_text, xc_text, yc_text, w_text, h_text = line.split()
        class_id = int(float(class_id_text))
        xc = float(xc_text) * image_width
        yc = float(yc_text) * image_height
        bw = float(w_text) * image_width
        bh = float(h_text) * image_height
        boxes.append(Box(class_id, xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
    return boxes


def grid_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """
    Generate sliding-window start positions along one image dimension.

    The final position is forced to touch the image border so no edge area is
    missed.
    """
    if length <= tile_size:
        return [0]

    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def make_tiles(width: int, height: int, tile_size: int, overlap: float) -> list[Tile]:
    """
    Create regular overlapping tiles that cover the whole image.

    The overlap value controls how much neighboring tiles share context.
    """
    if not 0 <= overlap < 1:
        raise ValueError("--overlap must be >= 0 and < 1")
    stride = max(1, int(round(tile_size * (1 - overlap))))
    xs = grid_positions(width, tile_size, stride)
    ys = grid_positions(height, tile_size, stride)
    return [
        Tile(x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in ys
        for x in xs
    ]


def make_center_tile(width: int, height: int, tile_size: int, center_x: float, center_y: float) -> Tile:
    """
    Create one tile centered near a target point and clamp it inside the image.

    This is used to make sure small Waldo boxes appear near the center of some
    training crops.
    """
    tile_width = min(tile_size, width)
    tile_height = min(tile_size, height)
    x1 = int(round(center_x - tile_width / 2))
    y1 = int(round(center_y - tile_height / 2))
    x1 = max(0, min(x1, width - tile_width))
    y1 = max(0, min(y1, height - tile_height))
    return Tile(x1, y1, x1 + tile_width, y1 + tile_height)


def make_center_tiles(
    width: int,
    height: int,
    boxes: list[Box],
    tile_size: int,
    jitter_count: int,
    jitter_scale: float,
    rng: random.Random,
) -> list[Tile]:
    """
    Create object-centered positive tiles plus jittered variants.

    Jitter adds nearby crops around each Waldo box so the model sees more
    location variation without losing the target.
    """
    tiles: list[Tile] = []
    for box in boxes:
        center_x = (box.x1 + box.x2) / 2
        center_y = (box.y1 + box.y2) / 2
        tiles.append(make_center_tile(width, height, tile_size, center_x, center_y))
        for _ in range(jitter_count):
            offset_x = rng.uniform(-tile_size * jitter_scale, tile_size * jitter_scale)
            offset_y = rng.uniform(-tile_size * jitter_scale, tile_size * jitter_scale)
            tiles.append(make_center_tile(width, height, tile_size, center_x + offset_x, center_y + offset_y))
    return tiles


def make_adaptive_center_tiles(
    width: int,
    height: int,
    boxes: list[Box],
    context: float,
    min_size: int,
    max_size: int,
    jitter_count: int,
    jitter_scale: float,
    rng: random.Random,
) -> list[Tile]:
    """
    Create object-centered tiles whose crop size depends on the target size.

    This helps very small boxes receive extra zoomed-in training examples while
    still keeping some surrounding context.
    """
    tiles: list[Tile] = []
    for box in boxes:
        box_width = max(1.0, box.x2 - box.x1)
        box_height = max(1.0, box.y2 - box.y1)
        crop_size = int(round(max(box_width, box_height) * context))
        crop_size = max(min_size, min(max_size, crop_size))
        center_x = (box.x1 + box.x2) / 2
        center_y = (box.y1 + box.y2) / 2
        tiles.append(make_center_tile(width, height, crop_size, center_x, center_y))
        for _ in range(jitter_count):
            offset_x = rng.uniform(-crop_size * jitter_scale, crop_size * jitter_scale)
            offset_y = rng.uniform(-crop_size * jitter_scale, crop_size * jitter_scale)
            tiles.append(make_center_tile(width, height, crop_size, center_x + offset_x, center_y + offset_y))
    return tiles


def clip_box_to_tile(box: Box, tile: Tile, min_visibility: float, min_box_px: float) -> Box | None:
    """
    Clip a full-image box to tile-local coordinates.

    Boxes are dropped if too little of the object remains visible or if the
    clipped box becomes too small to train reliably.
    """
    clipped = Box(
        class_id=box.class_id,
        x1=max(box.x1, tile.x1) - tile.x1,
        y1=max(box.y1, tile.y1) - tile.y1,
        x2=min(box.x2, tile.x2) - tile.x1,
        y2=min(box.y2, tile.y2) - tile.y1,
    )
    if clipped.area <= 0 or box.area <= 0:
        return None
    if clipped.area / box.area < min_visibility:
        return None
    if clipped.x2 - clipped.x1 < min_box_px or clipped.y2 - clipped.y1 < min_box_px:
        return None
    return clipped


def box_to_yolo(box: Box, tile: Tile) -> str:
    """Convert a tile-local pixel box back to normalized YOLO label format."""
    xc = ((box.x1 + box.x2) / 2) / tile.width
    yc = ((box.y1 + box.y2) / 2) / tile.height
    bw = (box.x2 - box.x1) / tile.width
    bh = (box.y2 - box.y1) / tile.height
    return f"{box.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def copy_or_crop_tile(image: Image.Image, tile: Tile, out_path: Path) -> None:
    """Crop one tile from the source image and save it as a JPEG file."""
    crop = image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
    crop.save(out_path, quality=95)


def clear_output_dir(path: Path) -> None:
    """Recreate the output dataset directory from scratch."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def make_split(
    src_image_dir: Path,
    src_label_dir: Path,
    dst_split_dir: Path,
    tile_size: int,
    overlap: float,
    min_visibility: float,
    min_box_px: float,
    negative_ratio: float,
    center_positives: bool,
    center_jitter: int,
    center_jitter_scale: float,
    adaptive_center_positives: bool,
    adaptive_context: float,
    adaptive_min_size: int,
    adaptive_max_size: int,
    rng: random.Random,
) -> tuple[int, int, int]:
    """
    Convert one dataset split into tiled images and labels.

    Positive tiles contain at least one Waldo box. Negative tiles are sampled
    background crops used to reduce false positives.
    """
    dst_image_dir = dst_split_dir / "images"
    dst_label_dir = dst_split_dir / "labels"
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    positive_tiles = 0
    negative_tiles = 0
    total_boxes = 0

    for image_path in image_files(src_image_dir):
        label_path = src_label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            boxes = read_yolo_boxes(label_path, width, height)
            tiles = make_tiles(width, height, tile_size, overlap)
            if center_positives and boxes:
                # Add extra crops centered on Waldo to strengthen positive examples.
                tiles.extend(
                    make_center_tiles(
                        width=width,
                        height=height,
                        boxes=boxes,
                        tile_size=tile_size,
                        jitter_count=center_jitter,
                        jitter_scale=center_jitter_scale,
                        rng=rng,
                    )
                )
                tiles = list(dict.fromkeys(tiles))
            if adaptive_center_positives and boxes:
                # Add zoomed crops whose size adapts to each Waldo bounding box.
                tiles.extend(
                    make_adaptive_center_tiles(
                        width=width,
                        height=height,
                        boxes=boxes,
                        context=adaptive_context,
                        min_size=adaptive_min_size,
                        max_size=adaptive_max_size,
                        jitter_count=center_jitter,
                        jitter_scale=center_jitter_scale,
                        rng=rng,
                    )
                )
                tiles = list(dict.fromkeys(tiles))

            candidates: list[tuple[Tile, list[Box]]] = []
            positives: list[tuple[Tile, list[Box]]] = []
            negatives: list[tuple[Tile, list[Box]]] = []

            for tile in tiles:
                # Keep only the labels that are sufficiently visible inside this tile.
                clipped_boxes = [
                    clipped
                    for box in boxes
                    if (clipped := clip_box_to_tile(box, tile, min_visibility, min_box_px)) is not None
                ]
                candidates.append((tile, clipped_boxes))
                if clipped_boxes:
                    positives.append((tile, clipped_boxes))
                else:
                    negatives.append((tile, clipped_boxes))

            keep: list[tuple[Tile, list[Box]]] = positives[:]
            if negative_ratio > 0 and negatives:
                # Limit the number of background tiles so the dataset is not dominated by negatives.
                if positives:
                    negative_count = min(len(negatives), math.ceil(len(positives) * negative_ratio))
                else:
                    negative_count = min(len(negatives), 1)
                keep.extend(rng.sample(negatives, negative_count))
            elif not positives and candidates:
                keep.append(rng.choice(candidates))

            for index, (tile, tile_boxes) in enumerate(keep):
                suffix = f"x{tile.x1}_y{tile.y1}_w{tile.width}_h{tile.height}"
                out_stem = f"{image_path.stem}__tile_{suffix}"
                out_image_path = dst_image_dir / f"{out_stem}.jpg"
                out_label_path = dst_label_dir / f"{out_stem}.txt"
                copy_or_crop_tile(image, tile, out_image_path)
                out_label_path.write_text(
                    "\n".join(box_to_yolo(box, tile) for box in tile_boxes),
                    encoding="utf-8",
                )
                if tile_boxes:
                    positive_tiles += 1
                    total_boxes += len(tile_boxes)
                else:
                    negative_tiles += 1

    return positive_tiles, negative_tiles, total_boxes


def main() -> None:
    """
    Command-line entry point for creating a tiled YOLO dataset.

    It reads the original `Data/data.yaml`, writes cropped images and adjusted
    labels, and creates a new data.yaml for training.
    """
    parser = argparse.ArgumentParser(description="Create a tiled YOLO dataset for small Waldo detection.")
    parser.add_argument("--data", default="Data/data.yaml", help="Source YOLO data.yaml.")
    parser.add_argument("--out", default="Data_tiled_768", help="Output tiled dataset directory.")
    parser.add_argument("--tile-size", type=int, default=768, help="Tile width/height in pixels.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio between neighboring tiles.")
    parser.add_argument("--min-visibility", type=float, default=0.35, help="Minimum original box area visible in a tile.")
    parser.add_argument("--min-box-px", type=float, default=3.0, help="Drop clipped boxes smaller than this many pixels.")
    parser.add_argument("--negative-ratio", type=float, default=3.0, help="Background tiles kept per positive tile.")
    parser.add_argument("--center-positives", action="store_true", help="Add object-centered positive tiles.")
    parser.add_argument(
        "--adaptive-center-positives",
        action="store_true",
        help="Add object-centered tiles with crop size based on each box size.",
    )
    parser.add_argument("--center-jitter", type=int, default=2, help="Extra jittered center tiles per box.")
    parser.add_argument(
        "--center-jitter-scale",
        type=float,
        default=0.15,
        help="Jitter range as a fraction of tile size for object-centered tiles.",
    )
    parser.add_argument("--adaptive-context", type=float, default=10.0, help="Adaptive crop size multiplier.")
    parser.add_argument("--adaptive-min-size", type=int, default=128, help="Smallest adaptive crop size.")
    parser.add_argument("--adaptive-max-size", type=int, default=512, help="Largest adaptive crop size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for negative tile sampling.")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    data = load_yaml(data_yaml)
    out_dir = Path(args.out)
    clear_output_dir(out_dir)

    rng = random.Random(args.seed)
    split_map = {"train": "train", "val": "valid", "test": "test"}
    for source_split, output_split in split_map.items():
        split_value = data.get(source_split)
        if not split_value:
            continue

        src_image_dir = split_path(data_yaml, str(split_value))
        src_label_dir = src_image_dir.parent / "labels"
        dst_split_dir = out_dir / output_split
        positives, negatives, boxes = make_split(
            src_image_dir=src_image_dir,
            src_label_dir=src_label_dir,
            dst_split_dir=dst_split_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_visibility=args.min_visibility,
            min_box_px=args.min_box_px,
            negative_ratio=args.negative_ratio,
            center_positives=args.center_positives,
            center_jitter=args.center_jitter,
            center_jitter_scale=args.center_jitter_scale,
            adaptive_center_positives=args.adaptive_center_positives,
            adaptive_context=args.adaptive_context,
            adaptive_min_size=args.adaptive_min_size,
            adaptive_max_size=args.adaptive_max_size,
            rng=rng,
        )
        print(
            f"{output_split}: {positives} positive tiles, {negatives} background tiles, "
            f"{boxes} boxes"
        )

    tiled_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": data["nc"],
        "names": data["names"],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(tiled_yaml, sort_keys=False), encoding="utf-8")
    print(f"\nWrote tiled dataset: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
