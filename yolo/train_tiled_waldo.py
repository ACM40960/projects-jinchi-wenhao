from __future__ import annotations # 

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    """
    Train YOLO on the tiled Waldo dataset.

    The tiled dataset makes the small Waldo target larger relative to each
    training image, which is the main improvement over the baseline script.
    """
    parser = argparse.ArgumentParser(description="Train YOLO on a tiled Waldo dataset.")
    parser.add_argument("--data", default="Data_tiled_416_center_neg05/data.yaml", help="Path to tiled YOLO dataset yaml.")
    parser.add_argument("--model", default="yolo26s.pt", help="Base model, for example yolo26s.pt or yolo26m.pt.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size matching the tile size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=4, help="Data loading workers.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="waldo_tiled_416_center_neg05_s", help="Experiment name.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted training run.")
    parser.add_argument("--lr0", type=float, default=0.005, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction.")
    parser.add_argument("--mosaic", type=float, default=0.7, help="Mosaic augmentation probability.")
    parser.add_argument("--close-mosaic", type=int, default=20, help="Disable mosaic for final N epochs.")
    parser.add_argument("--scale", type=float, default=0.35, help="Scale augmentation.")
    parser.add_argument("--translate", type=float, default=0.08, help="Translate augmentation.")
    parser.add_argument("--cache", action="store_true", help="Cache dataset images in RAM for faster training.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed Ultralytics training logs.")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain.")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL loss gain.")
    parser.add_argument("--cls", type=float, default=0.5, help="Classification loss gain.")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Tiled dataset yaml not found: {data_path}")

    project_path = Path(args.project).resolve()
    run_name = Path(args.name).name
    if run_name != args.name:
        raise ValueError("--name must be an experiment name only, not a path. Use --project for directories.")

    # Pass the tuned augmentation and loss settings into Ultralytics training.
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
        optimizer="auto",
        lr0=args.lr0,
        lrf=args.lrf,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        scale=args.scale,
        translate=args.translate,
        box=args.box,
        dfl=args.dfl,
        cls=args.cls,
        cache=args.cache,
        plots=True,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
