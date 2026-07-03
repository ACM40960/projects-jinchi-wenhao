from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    """
    Train the baseline YOLO model directly on the original full-size images.

    This is kept as the comparison experiment for the report. It uses
    `Data/data.yaml` by default and saves training outputs under `runs/detect`.
    """
    parser = argparse.ArgumentParser(description="Train a YOLO detect model to find Waldo.")
    parser.add_argument("--data", default="Data/data.yaml", help="Path to YOLO dataset yaml.")
    parser.add_argument("--model", default="yolo26n.pt", help="Base model, for example yolo26n.pt or yolo12n.pt.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Input image size. Larger helps small Waldo boxes.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=4, help="Data loading workers.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="waldo_yolo26n", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest interrupted training run.")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    project_path = Path(args.project).resolve()
    run_name = Path(args.name).name
    if run_name != args.name:
        raise ValueError("--name must be an experiment name only, not a path. Use --project for directories.")

    # Load the selected base checkpoint and start Ultralytics training.
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        task="detect",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=run_name,
        patience=args.patience,
        resume=args.resume,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
