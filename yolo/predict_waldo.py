from __future__ import annotations # 

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    """
    Run baseline YOLO prediction on full original images.

    This script is mainly useful for comparing the original-image method with
    the tiled inference method used by the final model.
    """
    parser = argparse.ArgumentParser(description="Predict Waldo locations with a trained YOLO detect model.")
    parser.add_argument("--weights", default="runs/detect/waldo_yolo26n/weights/best.pt", help="Trained model path.")
    parser.add_argument("--source", required=True, help="Image, folder, video, webcam index, or URL.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--project", default="runs/predict", help="Prediction output directory.")
    parser.add_argument("--name", default="waldo", help="Prediction run name.")
    parser.add_argument("--save-txt", action="store_true", help="Save predicted boxes as YOLO txt labels.")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped Waldo detections.")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found: {weights_path}. Run train_waldo.py first or pass --weights."
        )

    # Ultralytics handles image loading, NMS, saving visual results, and labels.
    model = YOLO(str(weights_path))
    results = model.predict(
        source=args.source, 
        task="detect",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
    )

    for result in results:
        # Print each predicted Waldo box so the command line output can be logged.
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print(f"{result.path}: no waldo detected")
            continue

        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            print(
                f"{result.path}: waldo #{index} "
                f"conf={conf:.3f} box_xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )


if __name__ == "__main__":
    main()
