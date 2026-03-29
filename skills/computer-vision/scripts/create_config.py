#!/usr/bin/env python3
"""
YOLO Hyperparameter YAML Creator

Creates a well-structured hyperparameter yaml file for a specific training run.
This yaml is the single source of truth for all modes (train/val/predict/export).

Usage:
    python create_config.py --model yolo11n --data datasets/pills --name pills-yolo11n
    python create_config.py --model yolo11m --data datasets/my_data --name my-experiment --epochs 200 --batch 8
"""

import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Prefer the env var set by app.py; fall back to resolving from __file__
# scripts/ → computer-vision/ → skills/ → <project_root>
_SKILL_SCRIPT_DIR = Path(__file__).resolve().parent          # …/skills/computer-vision/scripts
_FALLBACK_ROOT    = _SKILL_SCRIPT_DIR.parent.parent.parent   # …/vlm_skill
_PROJECT_ROOT_ENV = os.environ.get("PROJECT_ROOT")
if _PROJECT_ROOT_ENV and Path(_PROJECT_ROOT_ENV).exists():
    PROJECT_ROOT     = Path(_PROJECT_ROOT_ENV).resolve()
else:
    PROJECT_ROOT     = _FALLBACK_ROOT.resolve()

DEFAULT_CONFIGS = (PROJECT_ROOT / "results").resolve()
DEFAULT_WEIGHTS = (PROJECT_ROOT / "weights").resolve()
DEFAULT_DATASETS = (PROJECT_ROOT / "datasets").resolve()

# Ensure weights dir exists and redirect Ultralytics downloads there
DEFAULT_WEIGHTS.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(DEFAULT_WEIGHTS)
os.environ["ULTRALYTICS_DIR"] = str(DEFAULT_WEIGHTS)

try:
    from ultralytics import settings
    settings.update({
        "weights_dir": str(DEFAULT_WEIGHTS),
        "datasets_dir": str(DEFAULT_DATASETS),
        "runs_dir": str(DEFAULT_CONFIGS),
        "mlflow": False,
        "clearml": False,
        "tensorboard": False
    })
except ImportError:
    pass


TEMPLATE = {
    # ── Core Settings ──────────────────────────────────────────
    "task": "detect",           # (str) YOLO task: detect, segment, classify, pose, obb
    "mode": "train",            # (str) YOLO mode: train, val, predict, export, track, benchmark
    "model": "yolo11n.pt",      # (str) path to model file
    "data": None,               # (str) path to data file (e.g. coco8.yaml)
    "name": "my_experiment",    # (str) experiment name
    "project": None,            # (str) project name for results root

    # ── Train Settings ─────────────────────────────────────────
    "epochs": 100,              # (int) number of epochs to train for
    "time": None,               # (float) max hours to train; overrides epochs if set
    "patience": 100,            # (int) early stop after N epochs without improvement
    "batch": 16,                # (int | float) batch size or AutoBatch fraction
    "imgsz": 640,               # (int | list) train/val size
    "save": True,               # (bool) save checkpoints and results
    "save_period": -1,          # (int) save checkpoint every N epochs
    "cache": False,             # (bool | str) cache images in RAM or disk
    "device": None,             # (int | str | list) device: 0, [0,1], 'cpu', mps
    "workers": 0,               # (int) dataloader workers
    "exist_ok": False,          # (bool) overwrite existing experiment
    "pretrained": True,         # (bool | str) use pretrained weights
    "optimizer": "auto",        # (str) SGD, Adam, AdamW, RMSProp, etc.
    "verbose": True,            # (bool) print verbose logs
    "seed": 0,                  # (int) random seed
    "deterministic": True,      # (bool) enable deterministic ops
    "single_cls": False,        # (bool) treat all classes as a single class
    "rect": False,              # (bool) rectangular batches
    "cos_lr": False,            # (bool) cosine learning rate scheduler
    "close_mosaic": 10,         # (int) disable mosaic for final N epochs
    "resume": False,            # (bool) resume from last checkpoint
    "amp": True,                # (bool) Automatic Mixed Precision (AMP)
    "fraction": 1.0,            # (float) fraction of dataset to use
    "profile": False,           # (bool) profile ONNX/TensorRT speeds
    "freeze": None,             # (int | list) freeze first N layers
    "multi_scale": 0.0,         # (float) multi-scale range
    "overlap_mask": True,       # (bool) merge instance masks (segment only)
    "mask_ratio": 4,            # (int) mask downsample ratio (segment only)
    "dropout": 0.0,             # (float) dropout for classification (classify only)

    # ── Val/Test Settings ──────────────────────────────────────
    "val": True,                # (bool) run validation during training
    "split": "val",             # (str) split to evaluate: 'val', 'test', 'train'
    "save_json": False,         # (bool) save results to COCO JSON
    "save_hybrid": False,       # (bool) save hybrid labels
    "conf": None,               # (float) confidence threshold
    "iou": 0.7,                 # (float) IoU threshold for NMS
    "max_det": 300,             # (int) maximum number of detections
    "half": False,              # (bool) use half precision (FP16)
    "dnn": False,               # (bool) use OpenCV DNN for ONNX
    "plots": True,              # (bool) save plots and images
    "end2end": None,            # (bool) use end2end head (YOLO26, YOLOv10)

    # ── Predict Settings ──────────────────────────────────────
    "source": None,             # (str) path/dir/URL/stream for source
    "vid_stride": 1,            # (int) read every Nth frame for video
    "stream_buffer": False,     # (bool) buffer all frames for streams
    "visualize": False,         # (bool) visualize model features
    "augment": False,           # (bool) apply test-time augmentation
    "agnostic_nms": False,      # (bool) class-agnostic NMS
    "classes": None,            # (int | list) filter by class id(s)
    "retina_masks": False,      # (bool) use high-res segmentation masks
    "embed": None,              # (list) return feature embeddings

    # ── Visualize Settings ────────────────────────────────────
    "show": False,              # (bool) show images/videos window
    "save_frames": False,       # (bool) save individual frames
    "save_txt": False,          # (bool) save results as .txt
    "save_conf": False,         # (bool) save confidence scores
    "save_crop": False,         # (bool) save cropped regions
    "show_labels": True,        # (bool) draw labels
    "show_conf": True,          # (bool) draw confidence values
    "show_boxes": True,         # (bool) draw bounding boxes
    "line_width": None,         # (int) line width of boxes

    # ── Export Settings ───────────────────────────────────────
    "format": "onnx",           # (str) target format: onnx, openvino, engine, etc.
    "keras": False,             # (bool) enable Keras layers for TF
    "optimize": False,          # (bool) apply mobile optimizations
    "int8": False,              # (bool) INT8/PTQ quantization
    "dynamic": False,           # (bool) dynamic shapes
    "simplify": True,           # (bool) run ONNX graph simplifier
    "opset": None,              # (int) ONNX opset version
    "workspace": None,          # (float) TensorRT workspace size (GiB)
    "nms": False,               # (bool) fuse NMS into exported model

    # ── Hyperparameters ───────────────────────────────────────
    "lr0": 0.01,                # (float) initial learning rate
    "lrf": 0.01,                # (float) final LR fraction
    "momentum": 0.937,          # (float) SGD momentum or Adam beta1
    "weight_decay": 0.0005,     # (float) weight decay
    "warmup_epochs": 3.0,       # (float) warmup epochs
    "warmup_momentum": 0.8,     # (float) initial momentum during warmup
    "warmup_bias_lr": 0.1,      # (float) bias LR during warmup
    "box": 7.5,                 # (float) box loss gain
    "cls": 0.5,                 # (float) classification loss gain
    "dfl": 1.5,                 # (float) distribution focal loss gain
    "pose": 12.0,               # (float) pose loss gain (pose tasks)
    "kobj": 1.0,                # (float) keypoint objectness loss gain (pose tasks)
    "nbs": 64,                  # (int) nominal batch size
    "hsv_h": 0.015,             # (float) HSV hue augmentation
    "hsv_s": 0.7,               # (float) HSV saturation augmentation
    "hsv_v": 0.4,               # (float) HSV value augmentation
    "degrees": 0.0,             # (float) rotation degrees
    "translate": 0.1,           # (float) translation fraction
    "scale": 0.5,               # (float) scale gain
    "shear": 0.0,               # (float) shear degrees
    "perspective": 0.0,         # (float) perspective fraction
    "flipud": 0.0,              # (float) vertical flip probability
    "fliplr": 0.5,              # (float) horizontal flip probability
    "bgr": 0.0,                 # (float) RGB<->BGR swap probability
    "mosaic": 1.0,              # (float) mosaic probability
    "mixup": 0.0,               # (float) MixUp probability
    "copy_paste": 0.0,          # (float) copy-paste probability
    "auto_augment": "randaugment", # (str) auto augmentation policy
    "erasing": 0.4,             # (float) random erasing probability

    # ── Tracker settings ──────────────────────────────────────
    "tracker": "botsort.yaml",  # (str) botsort.yaml or bytetrack.yaml

}


def create_config(model: str, data: str, name: str = None, **overrides) -> dict:
    """Build a config dict from the template and overrides."""
    cfg = dict(TEMPLATE)

    # Handle environment variables and literal $PROJECT_ROOT
    data_str = os.path.expandvars(str(data))
    data_str = data_str.replace("$PROJECT_ROOT", str(PROJECT_ROOT))
    
    # Resolve data yaml to absolute path relative to PROJECT_ROOT
    data_path = Path(data_str)
    if not data_path.is_absolute() and not data_path.exists():
        # Only try PROJECT_ROOT resolution if it doesn't already exist locally
        # to allow for relative paths from CWD if they are valid
        test_path = PROJECT_ROOT / data_path
        if test_path.exists() or test_path.parent.exists():
            data_path = test_path
            
    data_path = data_path.resolve()
    if data_path.is_dir():
        yamls = list(data_path.glob("*.yaml"))
        cfg["data"] = str(yamls[0].resolve()) if yamls else str(data_path)
    elif not data_path.exists() and data_path.parent.exists():
        # Fallback if specified yaml doesn't exist but parent dir does
        yamls = list(data_path.parent.glob("*.yaml"))
        if yamls:
            print(f"[Info] '{data_path.name}' not found, using '{yamls[0].name}' instead.")
            cfg["data"] = str(yamls[0].resolve())
        else:
            cfg["data"] = str(data_path)
    else:
        cfg["data"] = str(data_path)

    # Resolve model to absolute path in weights directory
    model_name = model if model.endswith((".pt", ".yaml")) else model + ".pt"
    model_stem = Path(model_name).stem
    weights_path = (DEFAULT_WEIGHTS / model_name).resolve()
    cfg["model"] = str(weights_path)

    # Auto-generate name as [label/dataset]_[model]_[timestamp]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = data_path.parent.name if data_path.suffix == ".yaml" else data_path.name
    base_name = name if name else dataset_name
    cfg["name"] = f"{base_name}_{model_stem}_{ts}"

    # Apply any extra CLI overrides
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Create a YOLO hyperparameter YAML")
    parser.add_argument("--model",   required=True, help="Model name (e.g., yolo11n or yolo26m)")
    parser.add_argument("--data",    required=True, help="Path to dataset folder or data.yaml")
    parser.add_argument("--name",    default=None,  help="Optional label (default: auto [dataset]_[model])")
    parser.add_argument("--output",  default=None,  help="Custom output path for yaml file")
    
    # Standard overrides (added for better help message and type casting)
    parser.add_argument("--task",    default=None, choices=["detect","segment","classify","pose"])
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--imgsz",   type=int, default=None)
    parser.add_argument("--device",  default=None)
    
    # Parse known args first
    args, unknown = parser.parse_known_args()

    # Base overrides from standard arguments
    overrides = {k: v for k, v in vars(args).items() 
                 if k not in ("model", "data", "name", "output") and v is not None}

    # Parse dynamic overrides from unknown args (both --key=value and --key value)
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            if "=" in arg:
                k, v = arg[2:].split("=", 1)
                i += 1
            else:
                k = arg[2:]
                # If next token exists and is not a flag, it's the value. Otherwise, treat as boolean True.
                if i + 1 < len(unknown) and not unknown[i+1].startswith("--"):
                    v = unknown[i+1]
                    i += 2
                else:
                    v = True
                    i += 1
            
            # Try to cast to appropriate type based on TEMPLATE
            if k in TEMPLATE:
                orig_v = TEMPLATE[k]
                try:
                    if isinstance(orig_v, bool):
                        overrides[k] = str(v).lower() in ("true", "1", "t", "yes", "y")
                    elif isinstance(orig_v, int) and v is not True:
                        overrides[k] = int(v)
                    elif isinstance(orig_v, float) and v is not True:
                        overrides[k] = float(v)
                    else:
                        overrides[k] = v
                except ValueError:
                    overrides[k] = v # Fallback to string
            else:
                print(f"Warning: Unknown parameter '{k}' ignored.")
        else:
            i += 1

    cfg = create_config(args.model, args.data, args.name, **overrides)

    # Save the YAML alongside the results folder
    if args.output:
        output_path = Path(args.output)
    else:
        # Use the name (which now includes timestamp) for the folder
        result_dir = DEFAULT_CONFIGS / cfg["name"]
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / "hyperparameters.yaml"

    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    try:
        rel_path = output_path.relative_to(Path.cwd())
    except ValueError:
        rel_path = output_path

    print(f"[Config] Created: {rel_path}")
    print(f"[Next]   Run training with:")
    print(f"         python skills/computer-vision/scripts/vision_model_trainer.py --config {rel_path} --mode train")


if __name__ == "__main__":
    main()
