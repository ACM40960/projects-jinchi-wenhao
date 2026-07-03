# Waldo YOLO Detection

This project trains a YOLO object detection model to locate Waldo in images.

The dataset is already in Ultralytics YOLO detect format:

```text
class_id x_center y_center width height
```

`Data/data.yaml` defines one class:

```yaml
nc: 1
names: ['waldo']
```

## Install

This machine already has a working GPU environment named `waldo`.

Use it before running training:

```powershell
conda activate waldo
```

Or run commands without activating:

```powershell
conda run -n waldo python validate_dataset.py --data Data/data.yaml
```

If you need to install dependencies in a fresh environment:

```powershell
python -m pip install -r requirements.txt
```

## Check Dataset

```powershell
python validate_dataset.py --data Data/data.yaml
```

## Train

Default training uses the current Ultralytics YOLO nano detection model:

```powershell
python train_waldo.py --model yolo26n.pt --data Data/data.yaml --epochs 100 --imgsz 960 --batch 8
```

Without activating the environment:

```powershell
conda run -n waldo python train_waldo.py --model yolo26n.pt --data Data/data.yaml --epochs 100 --imgsz 960 --batch 8 --device 0
```

If you specifically want YOLO12:

```powershell
python train_waldo.py --model yolo12n.pt --name waldo_yolo12n --data Data/data.yaml --epochs 100 --imgsz 960 --batch 8
```

The best model is saved under:

```text
runs/detect/<run-name>/weights/best.pt
```

## Predict Waldo Location

For one image:

```powershell
python predict_waldo.py --weights runs/detect/waldo_yolo26n/weights/best.pt --source Data/test/images/waldo_img53.jpg
```

For a folder:

```powershell
python predict_waldo.py --weights runs/detect/waldo_yolo26n/weights/best.pt --source Data/test/images
```

The script prints Waldo positions as pixel boxes:

```text
box_xyxy=(left, top, right, bottom)
```

Annotated prediction images are saved in `runs/predict/waldo`.
