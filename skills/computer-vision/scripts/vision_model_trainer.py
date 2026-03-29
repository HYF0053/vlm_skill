#!/usr/bin/env python3
"""
YOLO Production Runner

Pure Python API runner for YOLO workflows.
Reads a hyperparameter YAML and runs the selected mode (train / val / predict / export).

Design principles:
  - All outputs are saved under: PROJECT_ROOT/results/[name]/
  - No CLI calls. Uses `from ultralytics import YOLO` directly.
  - Logs are filtered to epoch-level summaries only (no per-step spam).
  - Hyperparameter YAML is the single source of truth.

Usage:
    python vision_model_trainer.py --config path/to/hyperparameters.yaml --mode train
    python vision_model_trainer.py --config path/to/hyperparameters.yaml --mode val
    python vision_model_trainer.py --config path/to/hyperparameters.yaml --mode predict --source path/to/images
    python vision_model_trainer.py --config path/to/hyperparameters.yaml --mode export
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# ── Paths ─────────────────────────────────────────────────────────────────────
# Prefer the env var set by app.py; fall back to resolving from __file__
# scripts/ → computer-vision/ → skills/ → <project_root>
_SKILL_SCRIPT_DIR = Path(__file__).resolve().parent          # …/skills/computer-vision/scripts
_FALLBACK_ROOT    = _SKILL_SCRIPT_DIR.parent.parent.parent   # …/vlm_skill
_PROJECT_ROOT_ENV = os.environ.get("PROJECT_ROOT")
if _PROJECT_ROOT_ENV and Path(_PROJECT_ROOT_ENV).exists():
    PROJECT_ROOT     = Path(_PROJECT_ROOT_ENV).resolve()
else:
    PROJECT_ROOT     = _FALLBACK_ROOT.resolve()

DEFAULT_RESULTS  = (PROJECT_ROOT / "results").resolve()
DEFAULT_WEIGHTS  = (PROJECT_ROOT / "weights").resolve()
DEFAULT_DATASETS = (PROJECT_ROOT / "datasets").resolve()

# ── Force all Ultralytics auto-downloads into weights/ ───────────────────────
# This ensures .pt files never scatter to the project root or home directory
DEFAULT_WEIGHTS.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(DEFAULT_WEIGHTS)       # download cache dir
os.environ["ULTRALYTICS_DIR"]  = str(DEFAULT_WEIGHTS)      # model save dir

try:
    from ultralytics import settings
    settings.update({
        "weights_dir": str(DEFAULT_WEIGHTS),
        "datasets_dir": str(DEFAULT_DATASETS),
        "runs_dir": str(DEFAULT_RESULTS),
        "mlflow": False,
        "clearml": False,
        "tensorboard": False
    })
except ImportError:
    pass

# ── Logging: Only show important lines (suppress ultralytics step spam) ────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress ultralytics verbose step-by-step output in AI context
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a hyperparameter YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[Config] Loaded: {path}")

    # Validate dataset path robustness
    if "data" in config and config["data"]:
        data_str = os.path.expandvars(str(config["data"]))
        data_str = data_str.replace("$PROJECT_ROOT", str(PROJECT_ROOT))
        data_p = Path(data_str)
        
        if not data_p.is_absolute() and not data_p.exists():
            # Try resolving relative to DEFAULT_DATASETS or PROJECT_ROOT
            test_paths = [
                DEFAULT_DATASETS / data_p,
                PROJECT_ROOT / data_p,
                # If it's a filename only, maybe it's in datasets/
                DEFAULT_DATASETS / data_p.name
            ]
            for tp in test_paths:
                if tp.exists() or tp.parent.exists():
                    data_p = tp
                    break
        
        data_p = data_p.resolve()
        
        # Fallback if specified yaml doesn't exist but parent dir does
        if not data_p.exists() and data_p.parent.exists():
            yamls = list(data_p.parent.glob("*.yaml"))
            if yamls:
                print(f"[Path Fix] '{data_p.name}' not found, using '{yamls[0].name}' instead.")
                data_p = yamls[0]
                
        if data_p.exists():
            config["data"] = str(data_p)

    return config


def resolve_model(model_name: str) -> str:
    """Resolve model name to an absolute path in the weights/ folder."""
    p = Path(model_name)
    # Already an absolute path
    if p.is_absolute():
        return str(p)
    # Normalize extension
    if not model_name.endswith((".pt", ".yaml")):
        model_name += ".pt"
    weights_path = DEFAULT_WEIGHTS / model_name
    if weights_path.exists():
        print(f"[Weights] Found: {weights_path}")
    else:
        print(f"[Weights] '{model_name}' not in weights/. Will auto-download to {weights_path}.")
    return str(weights_path)


def append_result_log(result_dir: Path, entry: str):
    """Append a summary entry to the result log."""
    log_path = result_dir / "results_summary.txt"
    with open(log_path, "a") as f:
        f.write(f"--- {datetime.now().isoformat()} ---\n{entry}\n\n")
    print(f"[Log] Updated → {log_path}")


# ── Mode Runners ──────────────────────────────────────────────────────────────

def run_train(config: Dict[str, Any], config_path: str):
    """Run training using Python API based on the loaded config."""
    from ultralytics import YOLO

    model_name = config.get("model", "yolov11m.pt")
    # Use the directory where hyperparameters.yaml was found as the result folder
    result_dir = Path(config_path).resolve().parent

    # Build train kwargs from config, overriding project/name to our results dir
    train_args = {k: v for k, v in config.items() if k not in ("model", "name", "project")}
    train_args.update({
        "project": str(DEFAULT_RESULTS),
        "name":    result_dir.name,
        "exist_ok": True,
        "workers": config.get("workers", 0),
        "verbose": False,   # suppress step-level output
    })

    print(f"\n[Train] Starting → {result_dir}")
    model = YOLO(resolve_model(model_name))
    results = model.train(**train_args)

    # Extract and log epoch metrics
    summary = f"Model: {model_name}\nExperiment: {result_dir}\n"
    try:
        metrics = results.results_dict
        summary += "\n".join(f"  {k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float))
    except Exception:
        summary += "(Metrics not available)"
    
    append_result_log(result_dir, summary)
    print(f"\n[Train] Done. Results → {result_dir}")
    return result_dir


def run_val(config: Dict[str, Any], config_path: str):
    """Run validation using Python API."""
    from ultralytics import YOLO

    model_path = resolve_model(config.get("model", "best.pt"))
    # Use the directory where hyperparameters.yaml was found as the result folder
    result_dir = Path(config_path).resolve().parent

    print(f"\n[Val] Model: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(
        data=config.get("data"),
        imgsz=config.get("imgsz", 640),
        workers=config.get("workers", 0),
        project=str(result_dir.parent),
        name=result_dir.name,
        verbose=False,
    )

    summary = (
        f"Validation for: {model_path}\n"
        f"  mAP@50:    {metrics.box.map50:.4f}\n"
        f"  mAP@50:95: {metrics.box.map:.4f}\n"
        f"  Precision: {metrics.box.mp:.4f}\n"
        f"  Recall:    {metrics.box.mr:.4f}\n"
    )
    print(f"\n[Val Summary]\n{summary}")
    append_result_log(result_dir, summary)


def run_predict(config: Dict[str, Any], source: str, config_path: str):
    """Run prediction using Python API."""
    from ultralytics import YOLO

    model_path = resolve_model(config.get("model", "best.pt"))
    # Use the directory where hyperparameters.yaml was found as the result folder
    result_dir = Path(config_path).resolve().parent

    print(f"\n[Predict] Model: {model_path}, Source: {source}")
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        imgsz=config.get("imgsz", 640),
        conf=config.get("conf", 0.25),
        iou=config.get("iou", 0.7),
        save=True,
        project=str(result_dir),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )
    
    summary = f"Predict on '{source}': {len(results)} images processed → {result_dir}"
    print(f"\n[Predict] {summary}")
    append_result_log(DEFAULT_RESULTS / exp_name, summary)


def run_export(config: Dict[str, Any], config_path: str, dynamic: bool = True):
    """Export model using Python API."""
    from ultralytics import YOLO

    model_path  = resolve_model(config.get("model", "best.pt"))
    export_fmt  = config.get("format", "onnx")
    # Use the directory where hyperparameters.yaml was found as the result folder
    result_dir  = Path(config_path).resolve().parent

    print(f"\n[Export] Model: {model_path}, Format: {export_fmt}, Dynamic: {dynamic}")
    model = YOLO(model_path)
    exported = model.export(
        format=export_fmt,
        imgsz=config.get("imgsz", 640),
        half=config.get("half", False),
        simplify=config.get("simplify", True),
        dynamic=dynamic,
    )
    
    summary = f"Exported '{model_path}' → {exported} (Dynamic: {dynamic})"
    print(f"\n[Export] {summary}")
    append_result_log(result_dir, summary)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Python API Runner")
    parser.add_argument("--config",  required=True, help="Path to hyperparameters.yaml")
    parser.add_argument("--mode",    required=True, choices=["train", "val", "predict", "export"])
    parser.add_argument("--source",  default=None,  help="Source for predict mode (image/video/dir)")
    parser.add_argument("--dynamic", action="store_true", dest="dynamic", default=True, help="Enable dynamic batching for export (default: True)")
    parser.add_argument("--no-dynamic", action="store_false", dest="dynamic", help="Disable dynamic batching")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "train":
        run_train(config, args.config)
    elif args.mode == "val":
        run_val(config, args.config)
    elif args.mode == "predict":
        if not args.source:
            print("ERROR: --source is required for predict mode")
            sys.exit(1)
        run_predict(config, args.source, args.config)
    elif args.mode == "export":
        run_export(config, args.config, dynamic=args.dynamic)


if __name__ == "__main__":
    main()
