# Ultralytics YOLO Configuration Reference

Comprehensive guide to all configuration parameters for Training, Prediction, Validation, Export, Solutions, and Augmentation in the Ultralytics YOLO framework (v8-v12).

---

## 1. Train Settings
Used to configure the training process for YOLO models.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | str | `None` | Path to model file (e.g., yolov11n.pt, yolov11n.yaml). |
| `data` | str | `None` | Path to dataset config file (e.g., coco8.yaml). |
| `epochs` | int | `100` | Total number of training epochs. |
| `time` | float | `None` | Maximum training time in hours (overrides epochs). |
| `patience` | int | `100` | Epochs to wait for improvement before early stopping. |
| `batch` | int | `16` | Batch size (-1 for AutoBatch 60% GPU). |
| `imgsz` | int | `640` | Target image size for training. |
| `save` | bool | `True` | Save checkpoints and final weights. |
| `save_period` | int | `-1` | Save checkpoint every N epochs (-1 to disable). |
| `cache` | bool | `False` | Cache images in RAM (True/ram) or on disk (disk). |
| `device` | str/list | `None` | Device(s): `0`, `0,1,2,3`, `cpu`, `mps`. |
| `workers` | int | `0` | Number of worker threads for data loading (set to `0` for best stability on Windows). |
| `project` | str | `None` | Project directory name. |
| `name` | str | `None` | Experiment name. |
| `exist_ok` | bool | `False` | Allow overwriting existing experiment. |
| `pretrained` | bool/str| `True` | Start from pretrained model. |
| `optimizer` | str | `'auto'` | SGD, Adam, AdamW, NAdam, RAdam, RMSProp. |
| `seed` | int | `0` | Random seed for reproducibility. |
| `deterministic`| bool | `True` | Force deterministic algorithms. |
| `single_cls` | bool | `False` | Train multi-class data as a single class. |
| `rect` | bool | `False` | Rectangular training (min padding). |
| `cos_lr` | bool | `False` | Use cosine learning rate scheduler. |
| `close_mosaic` | int | `10` | Disable mosaic augmentation in last N epochs. |
| `resume` | bool | `False` | Resume training from last checkpoint. |
| `amp` | bool | `True` | Automatic Mixed Precision (AMP) training. |
| `fraction` | float | `1.0` | Fraction of dataset to use for training. |
| `lr0` | float | `0.01` | Initial learning rate. |
| `lrf` | float | `0.01` | Final learning rate fraction. |
| `momentum` | float | `0.937` | Momentum / beta1. |
| `weight_decay` | float | `0.0005` | L2 regularization. |
| `warmup_epochs`| float | `3.0` | LR warmup epochs. |
| `box` | float | `7.5` | Box loss weight. |
| `cls` | float | `0.5` | Classification loss weight. |
| `dfl` | float | `1.5` | Distribution Focal Loss (DFL) weight. |

---

## 2. Predict Settings
Settings influencing performance and accuracy during inference.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | str | `'ultralytics/assets'` | Source (image, video, dir, URL, digit for webcam). |
| `conf` | float | `0.25` | Confidence threshold. |
| `iou` | float | `0.7` | NMS IoU threshold. |
| `imgsz` | int/tuple| `640` | Inference image size. |
| `half` | bool | `False` | Use FP16 inference. |
| `device` | str | `None` | Inference device (cpu, cuda:0). |
| `max_det` | int | `300` | Maximum detections per image. |
| `vid_stride` | int | `1` | Frame stride for video inputs. |
| `stream_buffer`| bool | `False` | Buffer streaming frames. |
| `visualize` | bool | `False` | Visualize model features. |
| `augment` | bool | `False` | Test-time augmentation (TTA). |
| `agnostic_nms` | bool | `False` | Class-agnostic NMS. |
| `classes` | list[int] | `None` | Filter by class IDs. |
| `retina_masks` | bool | `False` | High-res segmentation masks. |
| `show` | bool | `False` | Show results in window. |
| `save` | bool | `False` | Save annotated images/videos. |
| `save_txt` | bool | `False` | Save results to .txt. |
| `save_conf` | bool | `False` | Include confidence in .txt. |
| `save_crop` | bool | `False` | Save cropped detections. |
| `show_labels` | bool | `True` | Show object labels. |
| `show_conf` | bool | `True` | Show confidence scores. |
| `show_boxes` | bool | `True` | Show bounding boxes. |

---

## 3. Validation Settings
Parameters for evaluating model performance.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | str | `None` | Dataset config file. |
| `imgsz` | int | `640` | Validation image size. |
| `batch` | int | `16` | Images per batch. |
| `save_json` | bool | `False` | Save results to JSON (COCO format). |
| `conf` | float | `0.001` | Confidence threshold (low for metrics). |
| `iou` | float | `0.7` | NMS IoU threshold. |
| `max_det` | int | `300` | Max detections per image. |
| `half` | bool | `False` | FP16 computation. |
| `device` | str | `None` | Validation device. |
| `plots` | bool | `True` | Save plots (F1, PR, confusion matrix). |
| `rect` | bool | `True` | Rectangular inference. |
| `split` | str | `'val'` | Dataset split (`val`, `test`). |

---

## 4. Export Settings
Configuration for converting models to production formats.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `format` | str | `'torchscript'` | Target: `onnx`, `engine`, `openvino`, `tflite`, `coreml`. |
| `imgsz` | int/tuple| `640` | Input size (640 or (h, w)). |
| `keras` | bool | `False` | Keras format (TensorFlow). |
| `optimize` | bool | `False` | TorchScript optimization for mobile. |
| `half` | bool | `False` | FP16 quantization. |
| `int8` | bool | `False` | INT8 quantization. |
| `dynamic` | bool | `False` | Dynamic input axes. |
| `simplify` | bool | `True` | ONNX graph simplification. |
| `opset` | int | `None` | ONNX opset version. |
| `workspace` | float | `None` | TensorRT workspace size (GiB). |
| `nms` | bool | `False` | Add NMS to exported model. |

---

## 5. Augmentation Settings
Hyperparameters for data augmentation during training.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `hsv_h` | float | `0.015` | Hue augmentation (fraction). |
| `hsv_s` | float | `0.7` | Saturation (fraction). |
| `hsv_v` | float | `0.4` | Value/Brightness (fraction). |
| `degrees` | float | `0.0` | Rotation (+/- deg). |
| `translate` | float | `0.1` | Translation (+/- fraction). |
| `scale` | float | `0.5` | Scale (+/- gain). |
| `shear` | float | `0.0` | Shear (+/- deg). |
| `perspective`| float | `0.0` | Perspective (+/- fraction). |
| `flipud` | float | `0.0` | Vertical flip (prob). |
| `fliplr` | float | `0.5` | Horizontal flip (prob). |
| `mosaic` | float | `1.0` | Mosaic augmentation (prob). |
| `mixup` | float | `0.0` | Mixup augmentation (prob). |
| `copy_paste` | float | `0.0` | Copy-paste augmentation (prob). |
| `auto_augment`| str | `'randaugment'`| Policy for classification. |
| `erasing` | float | `0.4` | Random erasing (prob). |

---

## 6. Solutions Settings
Parameters for Ultralytics specialized solutions (Counting, Heatmaps, etc.).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `region` | list | `[(20, 400), ...]` | Boundary points for counting. |
| `show_in` | bool | `True` | Display "in" counts. |
| `show_out` | bool | `True` | Display "out" counts. |
| `colormap` | int | `cv2.COLORMAP_JET` | Heatmap colormap. |
| `blur_ratio` | float | `0.5` | Blur intensity (0.1 - 1.0). |
| `records` | int | `5` | Count threshold to trigger alarm. |
| `vision_point`| tuple | `(20, 20)` | Point for VisionEye tracking. |
| `fps` | float | `30.0` | FPS for speed calculations. |
| `max_speed` | int | `120` | Max speed limit in overlays. |

---

For more details, visit the [Ultralytics Configuration Documentation](https://docs.ultralytics.com/usage/cfg/).
