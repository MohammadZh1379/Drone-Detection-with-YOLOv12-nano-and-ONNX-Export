# Drone-Detection-with-YOLOv12-nano-and-ONNX-Export
Binary drone detection (drone vs. background) using a single-class YOLOv12-nano detector.
# Drone Detection with YOLOv12-nano

Single-class drone (UAV) detection using a fine-tuned **YOLOv12-nano** model, trained on Kaggle (Tesla T4) with the Ultralytics framework and exported to **ONNX** for lightweight deployment.

## Overview

This project fine-tunes the pretrained `yolo12n.pt` checkpoint on a drone detection dataset. The pipeline covers dataset inspection, training, validation, a test prediction, and ONNX export. Everything is contained in a single notebook: `yolodrone.ipynb`.

## Dataset

- **Source:** [YOLO Drone Detection Dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset) by `muki2003` on Kaggle
- **Format:** YOLO (one `.txt` label file per image: `class x_center y_center width height`, normalized)
- **Classes:** 1 (`drone`)

| Split | Images |
|-------|--------|
| Train | 1,012 (1,011 used; 1 corrupt image was skipped) |
| Validation | 347 (369 drone instances) |

Dataset config used for training (`data.yaml`):

```yaml
path: /kaggle/input/datasets/muki2003/yolo-drone-detection-dataset/drone_dataset
train: train/images
val: valid/images
nc: 1
names:
  0: drone
```

## Training Setup

| Parameter | Value |
|-----------|-------|
| Model | YOLOv12n (`yolo12n.pt`, pretrained) |
| Epochs | 20 |
| Image size | 416 |
| Batch size | 16 |
| Optimizer | AdamW (auto-selected, lr = 0.002) |
| Hardware | Tesla T4 GPU (Kaggle) |
| Training time | about 5.6 minutes (0.093 h) |

## Results

Evaluation of `best.pt` on the validation set (347 images):

| Metric | Value |
|--------|-------|
| Precision | 0.923 |
| Recall | 0.791 |
| mAP@50 | 0.885 |
| mAP@50-95 | 0.544 |

Model size: 2.56M parameters, 7.3 GFLOPs (fused). Validation inference speed on a T4 was about 3.6 ms per image at 416 px.

Training progressed from mAP@50 = 0.51 after the first epoch to about 0.89 by epochs 18-20.

## Export

The best checkpoint is exported to ONNX (opset 12, input 416x416, simplified):

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")
model.export(format="onnx", imgsz=416, simplify=True, opset=12)
```

The resulting `best.onnx` file is about 10 MB and can be used with ONNX Runtime or any ONNX-compatible inference engine.

## Getting Started

### Requirements

```bash
pip install ultralytics kagglehub opencv-python matplotlib numpy torch
```

### Run the notebook

1. Open `yolodrone.ipynb` on Kaggle (recommended, with a GPU accelerator) or locally.
2. Download the dataset with `kagglehub` (the first cell does this automatically).
3. Update the dataset paths in `data.yaml` if you are not running on Kaggle.
4. Run the cells in order: dataset preview, training, validation, prediction, ONNX export.

### Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")  # or "best.onnx"
results = model.predict(source="image.jpg", imgsz=416, conf=0.5, save=True)
print(results[0].boxes)
```

## Project Structure

```
.
├── yolodrone.ipynb   # Full pipeline: data preview, training, validation, export
└── README.md
```

## Notes and Limitations

- The model detects a single class (`drone`) and was trained on a relatively small dataset, so performance may drop on unseen environments, very small or distant drones, or unusual lighting.
- Recall (0.79) is lower than precision (0.92), meaning some drones are missed. More epochs, a larger input size, or more data may improve this.
- In a sanity check on an unrelated image (a car photo) at `conf=0.5`, the model produced no detections, as expected.

## Author

**Mohammad Sadra Zohouri**
