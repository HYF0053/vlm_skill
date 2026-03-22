import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAMSegmenter:
    """
    SAM 2 based instance segmenter. 
    Refines BBoxes into masks for instance segmentation datasets.
    """
    def __init__(self, model_cfg: str, checkpoint: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

    def process_image(self, image_path: str, bboxes: list):
        """
        Produce masks from a list of BBoxes for a single image.
        bboxes: list of [ymin, xmin, ymax, xmax] pixel coordinates.
        """
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        self.predictor.set_image(image_np)

        # Convert [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax] for SAM
        input_boxes = np.array([[b[1], b[0], b[3], b[2]] for b in bboxes])

        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # masks shape: (N_boxes, 1, H, W)
        return masks, scores

def rle_encode(mask):
    """Simple RLE encoding for saving masks in JSON efficiently."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main():
    parser = argparse.ArgumentParser(description="SAM 2 Precision Segmenter")
    parser.add_argument("labels_json", type=str, help="Output from vlm_labeler.py")
    parser.add_argument("--model-cfg", type=str, required=True, help="SAM 2 config file (.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="SAM 2 checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="segment_labels.json", help="Final output file")
    
    args = parser.parse_args()

    # Load data
    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segmenter = SAMSegmenter(args.model_cfg, args.checkpoint)
    
    final_output = {}

    for img_name, info in data.items():
        img_path = info["image_path"]
        detections = info["detections"]
        
        if not detections:
            continue
            
        pixel_boxes = [d["pixel_box"] for d in detections]
        print(f"Segmenting {img_name} ({len(pixel_boxes)} masks)...")
        
        masks, scores = segmenter.process_image(img_path, pixel_boxes)
        
        # Package masks back into JSON
        for i, detection in enumerate(detections):
            # Encode mask as RLE or polygon to avoid huge JSON files
            # Here we just save the 1/0 mask encoded as RLE for demonstration
            mask_bool = masks[i, 0] # (H, W)
            detection["mask_rle"] = rle_encode(mask_bool)
            detection["mask_score"] = float(scores[i, 0])
            
        final_output[img_name] = info

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    
    print(f"Segmented labels saved to {args.output}")

if __name__ == "__main__":
    main()
