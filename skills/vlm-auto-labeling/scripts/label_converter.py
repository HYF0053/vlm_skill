import json
import argparse
import os
from pathlib import Path
from PIL import Image

def convert_to_yolo(data: dict, output_dir: str):
    """
    Convert Silver JSON labels (from VLM-labeling) to YOLO format (one .txt per image).
    Normalized center_x, center_y, width, height (0-1).
    """
    output_path = Path(output_dir)
    labels_path = output_path / "labels"
    labels_path.mkdir(parents=True, exist_ok=True)
    
    classes = []
    
    for img_name, info in data.items():
        img = Image.open(info["image_path"])
        w, h = img.size
        
        yolo_lines = []
        for det in info["detections"]:
            # Check if mask_score is high enough (optional filtering)
            if det.get("mask_score", 1.0) < 0.5:
                 continue
                 
            label = det["label"]
            if label not in classes:
                classes.append(label)
            
            cls_id = classes.index(label)
            
            # Pixel BBox [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = det["pixel_box"]
            
            # YOLO format: cls_id, x_center, y_center, width, height (normalized 0-1)
            dw = 1.0 / w
            dh = 1.0 / h
            x_center = (xmin + xmax) / 2.0 * dw
            y_center = (ymin + ymax) / 2.0 * dh
            width = (xmax - xmin) * dw
            height = (ymax - ymin) * dh
            
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
        # Write .txt file (matches image filename)
        txt_name = Path(img_name).stem + ".txt"
        with open(labels_path / txt_name, "w") as f:
            f.write("\n".join(yolo_lines))
            
    # Write classes.txt
    with open(output_path / "classes.txt", "w") as f:
        f.write("\n".join(classes))
    
    # Generate data.yaml (simplified for YOLOv8)
    with open(output_path / "data.yaml", "w") as f:
        f.write(f"train: {os.path.abspath(output_path / 'images')}/train\n")
        f.write(f"val: {os.path.abspath(output_path / 'images')}/val\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")

    print(f"YOLO labels generated in {labels_path}")
    print(f"Classes found: {classes}")

def main():
    parser = argparse.ArgumentParser(description="Multi-format Label Converter")
    parser.add_argument("labels_json", type=str, help="JSON produced by VLM/SAM workflow")
    parser.add_argument("--format", type=str, choices=["yolo", "coco"], default="yolo")
    parser.add_argument("--output", type=str, default="dataset_output", help="Output directory")
    
    args = parser.parse_args()

    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.format == "yolo":
        convert_to_yolo(data, args.output)
    elif args.format == "coco":
        # Implementation left as placeholder or for next iteration
        print("COCO support coming soon. Stick with YOLO for now!")

if __name__ == "__main__":
    main()
