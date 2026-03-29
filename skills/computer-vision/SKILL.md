---
name: "ultralytics-expert"
description: Computer vision engineering skill focused exclusively on the Ultralytics framework for object detection, instance segmentation, image classification, and pose estimation. Covers dataset preparation, training, and production deployment with ONNX/TensorRT.
---

# Ultralytics YOLO Expert

Production computer vision engineering skill focused on the Ultralytics YOLO framework (v7-v12). Optimized for rapid development, experiment tracking, and production-grade deployment.

## Table of Contents

- [Project Structure & Paths](#project-structure--paths)
- [Workflow 1: Dataset Engineering](#workflow-1-dataset-engineering)
- [Workflow 2: Training Pipeline](#workflow-2-training-pipeline)
- [Workflow 3: Validation & Prediction](#workflow-3-validation--prediction)
- [Workflow 4: Export & Optimization](#workflow-4-export--optimization)
- [YOLO Model Selection Guide (2026 Edition)](#yolo-model-selection-guide-2026-edition)
- [Supported Architecture Metadata](#supported-architecture-metadata)
- [Reference Documentation](#reference-documentation)

---

## Project Structure & Paths

All experiments and data are anchored to **`PROJECT_ROOT`**.

### Directory Layout
- **Datasets**: `$PROJECT_ROOT/datasets/[dataset_name]/`
- **Pretrained Weights**: `$PROJECT_ROOT/weights/` (managed via `ultralytics.settings`)
- **Results**: `$PROJECT_ROOT/results/[name]_[timestamp]/`
    - `hyperparameters.yaml`: Your input configuration.
    - `args.yaml`: The actual parameters used by the engine.
    - `weights/`: Best and last weights (.pt).

### Weight Management
To ensure a clean workspace, weights are automatically directed to the project's weights folder:
```python
from ultralytics import settings
settings.update({"weights_dir": "$PROJECT_ROOT/weights"})
```

### Strict Agent Protocol
1. **NO DIRECT CLI**: Do not use the `yolo` command directly (CLI is disabled in Docker).
2. **USE SKILL SCRIPTS**: Always use the provided Python scripts.
3. **UNIFIED FOLDERS**: All related files (config, models, logs) for a single run stay in the SAME timestamped folder.

---

## Workflow 1: Dataset Engineering

### Step 1: Audit and Clean
```bash
# Analyze raw data
python $PROJECT_ROOT/skills/computer-vision/scripts/dataset_pipeline_builder.py $PROJECT_ROOT/data/raw/ --analyze

# Clean corrupted files
python $PROJECT_ROOT/skills/computer-vision/scripts/dataset_pipeline_builder.py $PROJECT_ROOT/data/raw/ --clean --output $PROJECT_ROOT/data/cleaned/
```

### Step 2: Convert and Split
```bash
# Convert to YOLO format
python $PROJECT_ROOT/skills/computer-vision/scripts/dataset_pipeline_builder.py $PROJECT_ROOT/data/cleaned/ --format yolo --output $PROJECT_ROOT/datasets/my_dataset

# Split into train/val/test
python $PROJECT_ROOT/skills/computer-vision/scripts/dataset_pipeline_builder.py $PROJECT_ROOT/datasets/my_dataset --split 0.8 0.1 0.1 --output $PROJECT_ROOT/datasets/my_dataset
```

---

## Workflow 2: Training Pipeline

### Step 1: Create Config
Generate the timestamped experiment folder and the YAML blueprint.
```bash
python $PROJECT_ROOT/skills/computer-vision/scripts/create_config.py \
    --model yolo11n \
    --data $PROJECT_ROOT/datasets/my_dataset/data.yaml \
    --epochs 100 --batch 16
```
> **Note**: This will automatically create a folder like: `results/my_dataset_yolo11n_20260329_145500/`

### Step 2: Run Training
Pass the generated config file in the timestamped folder.
```bash
python $PROJECT_ROOT/skills/computer-vision/scripts/vision_model_trainer.py \
    --config $PROJECT_ROOT/results/my_dataset_yolo11n_20260329_145500/hyperparameters.yaml \
    --mode train
```
> **Note**: All outputs (weights, logs) will stay in this same directory.

---

## Workflow 3: Validation & Prediction

### Validation
```bash
python $PROJECT_ROOT/skills/computer-vision/scripts/vision_model_trainer.py \
    --config $PROJECT_ROOT/results/my_dataset_yolo11n_20260329_145500/hyperparameters.yaml \
    --mode val
```

### Prediction
```bash
python $PROJECT_ROOT/skills/computer-vision/scripts/vision_model_trainer.py \
    --config $PROJECT_ROOT/results/my_dataset_yolo11n_20260329_145500/hyperparameters.yaml \
    --mode predict \
    --source $PROJECT_ROOT/data/test_images/
```

---

## Workflow 4: Export & Optimization

### Export to ONNX/TensorRT
```bash
python $PROJECT_ROOT/skills/computer-vision/scripts/vision_model_trainer.py \
    --config $PROJECT_ROOT/results/my_dataset_yolo11n_20260329_145500/hyperparameters.yaml \
    --mode export --dynamic
```

### Inference Optimization

Use `inference_optimizer.py` to analyze, benchmark, and get optimization recommendations for your models (`.pt`, `.onnx`).

#### 1. Model Analysis (`--analyze`)
Analyze model structure, parameters, layers, and input/output shapes:
```bash
python $PROJECT_ROOT/skills/computer-vision/scripts/inference_optimizer.py \
    $PROJECT_ROOT/results/my_dataset_yolo11n_20260329_145500/weights/best.pt \
    --analyze
```

#### 2. Benchmarking (`--benchmark`)
Test inference speed, latency, and throughput across different batch sizes:
```bash
python $PROJECT_ROOT/skills/computer-vision/scripts/inference_optimizer.py \
    $PROJECT_ROOT/results/my_dataset_yolo11n_20260329_145500/weights/best.pt \
    --benchmark --input-size 640 640 --batch-sizes 1 4 8
```

#### 3. Optimization Recommendations (`--recommend`)
Get platform-specific acceleration steps and commands based on your deployment target (`gpu`, `cpu`, `edge`, `mobile`, `apple`, `intel`):
```bash
python $PROJECT_ROOT/skills/computer-vision/scripts/inference_optimizer.py \
    $PROJECT_ROOT/results/my_dataset_yolo11n_20260329_145500/weights/best.pt \
    --recommend --target gpu
```

#### Selection Guide Matrix

| Deployment Target | Optimization Path |
|-------------------|-------------------|
| NVIDIA GPU (cloud) | PyTorch → ONNX → TensorRT FP16 |
| NVIDIA GPU (edge) | PyTorch → TensorRT INT8 |
| Intel CPU | PyTorch → ONNX → OpenVINO |
| Apple Silicon | PyTorch → CoreML |
| Generic CPU | PyTorch → ONNX Runtime |
| Mobile | PyTorch → TFLite or ONNX Mobile |

---

## YOLO Model Selection Guide (2026 Edition)

| Series | Model | mAP | Params | T4 Latency | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO26** | `n` | 40.9 | 2.4M | 1.7ms | **LATEST** Next-gen production |
| (Ultralytics) | `s` | 48.6 | 9.5M | 2.5ms | Balanced speed/accuracy |
| | `m` | 53.1 | 20.4M | 4.7ms | High-performance detection |
| **YOLO12** | `n` | 40.6 | 2.6M | 1.6ms | Highly optimized ARM/Edge |
| (Tian et al.) | `s` | 48.0 | 9.3M | 2.6ms | Robust real-time apps |
| | `m` | 52.5 | 20.2M | 4.9ms | Advanced segmentation |
| **YOLO11** | `x` | 54.7 | 56.9M | 11.3ms | SOTA accuracy (2024) |
| **YOLOv10** | `m` | 51.3 | 15.4M | 5.5ms | Efficient NMS-free models |
| **YOLOv8** | `m` | 50.2 | 25.9M | 5.9ms | Legacy stable production |

*Note: Latency measured at 640x imgsz on NVIDIA T4 GPU.*

---

## Supported Architecture Metadata

| Model | Author/Org | Date | Key Strength |
| :--- | :--- | :--- | :--- |
| **YOLO26** | Ultralytics | 2026 | Enhanced feature extraction & speed |
| **YOLO12** | Buffalo/UCAS | 2025 | Structural re-parameterization |
| **YOLO11** | Ultralytics | 2024 | Versatility (Seg/Pose/OBB) |
| **YOLOv10** | Tsinghua | 2024 | Zero-NMS for low latency |
| **YOLOv9** | Academia Sinica | 2024 | Programmable Gradient Information |
| **YOLOv7** | Academia Sinica | 2022 | Concatenation-based architecture |

---

## Reference Documentation

- [Computer Vision Architectures](references/computer_vision_architectures.md)
- [Object Detection Optimization](references/object_detection_optimization.md)
- [Config Reference](references/yolo_config_reference.md)

## Resources
- **Trainer Script**: `scripts/vision_model_trainer.py`
- **Dataset Tool**: `scripts/dataset_pipeline_builder.py`
- **Optimizer**: `scripts/inference_optimizer.py`
