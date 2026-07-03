from __future__ import annotations # 

import argparse
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Missing dependency: pyyaml. Install requirements first: pip install -r requirements.txt") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path: Path) -> dict:
    """
    Load and validate the top-level YOLO data.yaml file.

    The returned dictionary should contain split paths, class count, and names.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset yaml: {path}")
    return data


def split_dir(data_yaml: Path, split_value: str) -> Path:
    """Resolve train/val/test image directories relative to data.yaml."""
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return data_yaml.parent / split_path


def validate_label_file(label_path: Path, nc: int, allow_empty: bool) -> tuple[list[str], list[str]]:
    """
    Validate one YOLO label file.

    Checks class id range, normalized coordinates, box width/height, and whether
    empty label files are allowed for background images.
    """
    errors: list[str] = []
    warnings: list[str] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        message = f"{label_path}: empty label file, treated as a background image"
        if allow_empty:
            warnings.append(message)
        else:
            errors.append(message)
        return errors, warnings

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no}: expected 5 values, got {len(parts)}")
            continue

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_no}: non-numeric label values")
            continue

        if class_id < 0 or class_id >= nc:
            errors.append(f"{label_path}:{line_no}: class id {class_id} outside range 0..{nc - 1}")

        for name, value in (
            ("x_center", x_center),
            ("y_center", y_center),
            ("width", width),
            ("height", height),
        ):
            if value < 0 or value > 1:
                errors.append(f"{label_path}:{line_no}: {name}={value} is not normalized to 0..1")

        if width <= 0 or height <= 0:
            errors.append(f"{label_path}:{line_no}: width and height must be > 0")

    return errors, warnings


def validate_split(
    data_yaml: Path,
    split_name: str,
    split_value: str,
    nc: int,
    allow_empty: bool,
) -> tuple[int, int, list[str], list[str]]:
    """
    Validate image-label pairing and label contents for one dataset split.

    Returns image count, label count, errors, and warnings.
    """
    image_dir = split_dir(data_yaml, split_value)
    label_dir = image_dir.parent / "labels"
    errors: list[str] = []
    warnings: list[str] = []

    if not image_dir.exists():
        return 0, 0, [f"{split_name}: image directory not found: {image_dir}"], warnings
    if not label_dir.exists():
        return 0, 0, [f"{split_name}: label directory not found: {label_dir}"], warnings

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    labels = sorted(label_dir.glob("*.txt"))
    label_stems = {path.stem for path in labels}
    image_stems = {path.stem for path in images}

    for image in images:
        if image.stem not in label_stems:
            errors.append(f"{split_name}: missing label for image {image}")

    for label in labels:
        if label.stem not in image_stems:
            errors.append(f"{split_name}: missing image for label {label}")
        label_errors, label_warnings = validate_label_file(label, nc, allow_empty)
        errors.extend(label_errors)
        warnings.extend(label_warnings)

    return len(images), len(labels), errors, warnings


def main() -> None:
    """
    Command-line entry point for checking a YOLO detection dataset.

    This is useful before training to catch missing files or malformed labels.
    """
    parser = argparse.ArgumentParser(description="Validate a YOLO detect dataset for Waldo training.")
    parser.add_argument("--data", default="Data/data.yaml", help="Path to dataset yaml.")
    parser.add_argument(
        "--no-empty-labels",
        action="store_true",
        help="Fail validation if an image has an empty label file.",
    )
    args = parser.parse_args()

    data_yaml = Path(args.data)
    data = load_data_yaml(data_yaml)
    nc = int(data.get("nc", 0))
    if nc <= 0:
        raise ValueError("data.yaml must define a positive nc value.")

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for split_name in ("train", "val", "test"):
        split_value = data.get(split_name)
        if not split_value:
            print(f"{split_name}: not configured, skipped")
            continue
        image_count, label_count, errors, warnings = validate_split(
            data_yaml,
            split_name,
            str(split_value),
            nc,
            allow_empty=not args.no_empty_labels,
        )
        print(f"{split_name}: {image_count} images, {label_count} labels")
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    if all_warnings:
        print("\nDataset warnings:")
        for warning in all_warnings[:100]:
            print(f"- {warning}")
        if len(all_warnings) > 100:
            print(f"- ... {len(all_warnings) - 100} more warnings")

    if all_errors:
        print("\nDataset validation failed:")
        for error in all_errors[:100]:
            print(f"- {error}")
        if len(all_errors) > 100:
            print(f"- ... {len(all_errors) - 100} more errors")
        raise SystemExit(1)

    print("\nDataset validation passed. Format is YOLO detect: class x_center y_center width height.")


if __name__ == "__main__":
    main()
