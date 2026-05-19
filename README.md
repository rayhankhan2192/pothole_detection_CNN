```markdown
# Road Damage Classification & Object Detection Pipeline

An end-to-end computer vision framework developed in PyTorch and Ultralytics YOLOv8 to automatically identify, classify, and localize road surface distresses: **Cracks**, **Potholes**, and **Surface Erosion**.

---

## Project Structure
Ensure your local project directory is organized as follows before running the scripts:

```bash
pothole_detection_CNN/
│
├── models/
│   ├── __init__.py
│   ├── cnn_model.py          # Custom CNN without MLP
│   ├── efficientnet_model.py # Frozen EfficientNetB0
│   ├── mobilenet_model.py    # Frozen MobileNetV2
│   ├── resnet_model.py       # Frozen ResNet50
│   ├── vgg_model.py          # Frozen VGG19
│   └── model_factory.py      # Model loading, Focal Loss, Label Smoothing
│
├── utils/
│   ├── __init__.py
│   ├── dataloader.py         # RoadDamageDataset loader & Albumentations pipelines
│   └── train_evaluation.py   # Trainer class with early stopping & custom plots
│
├── run.py                    # Main classification pipeline orchestrator
├── detection.py              # Main YOLOv8 object detection training script
└── README.md

```

---

## Part 1: Road Damage Classification (PyTorch)

This pipeline processes RGB images into 3 distinct damage classes. It automatically handles data splitting (80/10/10), dynamic class weight balancing for imbalanced data, and tracks training history with automated logging and plot generation.

### Dataset Setup

Your classification raw dataset must be structured by folder names matching the categories:

```bash
Dataset/
├── Crack/
│   ├── img1.jpg
│   └── ...
├── Pothole/
│   ├── img1.jpg
│   └── ...
└── Surface erosion/
    ├── img1.jpg
    └── ...

```

### How to Run Classification

Run `run.py` from your terminal inside your virtual environment (`venv`).

#### 1. Baseline Training (Custom CNN, Standard Augmentation)

```bash
python run.py --data-dir "path/to/your/Dataset" --model-name customcnn --epochs 30 --batch-size 32

```

#### 2. Advanced Training (ResNet50, Enhanced Augmentation, Focal Loss)

Use the `enhanced` augmentation strategy (which applies **CLAHE** and **Unsharp Masking**) combined with **Focal Loss** to fight extreme class imbalances:

```bash
python run.py --data-dir "path/to/your/Dataset" --model-name resnet50 --aug-type enhanced --loss focal --epochs 50 --lr 0.0005

```

### Arguments Reference

| Argument | Type | Default | Choices | Description |
| --- | --- | --- | --- | --- |
| `--data-dir` | `str` | *Required* | - | Path to the root directory containing class folders |
| `--model-name` | `str` | `customcnn` | `customcnn`, `efficientnet`, `mobilenet`, `resnet50`, `vgg19` | Network architecture (all transfer learning models have **No MLP**) |
| `--batch-size` | `int` | `32` | - | Mini-batch training size |
| `--aug-type` | `str` | `standard` | `none`, `standard`, `enhanced` | Data augmentation pipeline |
| `--loss` | `str` | `crossentropy` | `crossentropy`, `focal`, `labelsmoothing` | Loss objective function |
| `--epochs` | `int` | `30` | - | Maximum training epochs (monitored by Early Stopping) |
| `--lr` | `float` | `0.001` | - | Initial learning rate |

---

## Part 2: Road Damage Object Detection (YOLOv8)

This pipeline handles bounding-box localization of road anomalies using the state-of-the-art **YOLOv8** framework.

### Dataset Setup

Your detection dataset must contain standard YOLO annotation files (`.txt`) organized by splits and be configured via a `data.yaml` file.

Your `data.yaml` structure should look like this:

```yaml
path: /absolute/path/to/your/dataset_folder  # Or relative path from script
train: images/train
val: images/val
test: images/test

names:
  0: Crack
  1: Pothole
  2: Surface erosion

```

### How to Run Detection

Run `detection.py` inside your environment.

> **Windows/Git Bash Terminal Note:** If your directory contains spaces (e.g., `Dataset/Data Y12 Final`), you **must enclose the path in single quotes** or escape the spaces using backslashes to prevent argument parsing errors.

#### 1. Baseline Detection Training (YOLOv8 Nano)

```bash
python detection.py --data 'Dataset/Data Y12 Final/data.yaml' --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640

```

#### 2. Advanced Detection Training (YOLOv8 Small with Focal Loss & AMP Enabled)

Activate **Focal Loss** to penalize misclassifications on rare road distress classes and use **Automatic Mixed Precision (AMP)** for faster execution:

```bash
python detection.py --data 'Dataset/Data Y12 Final/data.yaml' --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640 --fl_gamma 1.5 --amp True

```

#### 3. Resume an Interrupted Training Run

If your system crashes or gets interrupted, seamlessly pick up from your last saved checkpoint:

```bash
python detection.py --data 'Dataset/Data Y12 Final/data.yaml' --model 'Result/yolov8s/weights/last.pt' --resume

```

### Arguments Reference

| Argument | Type | Default | Choices | Description |
| --- | --- | --- | --- | --- |
| `--model` | `str` | `yolov8n.pt` | - | Pre-trained weights name or path to a custom `.pt` checkpoint |
| `--data` | `str` | `data.yaml` | - | Path to your dataset configuration `data.yaml` file |
| `--epochs` | `int` | `None` | - | Total number of training cycles (YOLO defaults apply if omitted) |
| `--batch` | `int` | `None` | - | Training batch size override |
| `--imgsz` | `int` | `None` | - | Input image resolution dimensions (Default: `640`) |
| `--patience` | `int` | `None` | - | Early stopping patience epochs |
| `--optimizer` | `str` | `None` | `SGD`, `Adam`, `AdamW`, etc. | Optimization algorithm choice |
| `--amp` | `str` | `None` | `True`, `False` | Enable/Disable Automatic Mixed Precision safely |
| `--resume` | `flag` | `False` | - | Include flag to resume training from the last saved block |
| `--fl_gamma` | `float` | `0.0` | - | Focal Loss gamma setting for class imbalance (>0 to activate) |
| `--device` | `str` | `auto` | `auto`, `0`, `cpu`, `mps` | Hardware environment selection (`mps` for Apple Silicon) |

---

## Outputs and Evaluation Results

All historical metadata, weights, and graphs are safely structured inside the `Result/` repository output loop:

* **Classification Output:** Saved to `Result/<model_name>_<aug_type>/`. Contains training curve graphs, best weight checkpoints, and validation/test confusion matrices.
* **Detection Output:** Saved to `Result/<model_base_name>/`. Contains YOLO validation batches, precision-recall metrics curves, and final model weights (`best.pt`, `last.pt`).

```

```