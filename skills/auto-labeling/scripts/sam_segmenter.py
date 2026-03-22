import os
import json
import argparse
import numpy as np
from pathlib import Path
from ultralytics import SAM

class UltralyticsSAMSegmenter:
    """
    Ultralytics-based SAM / SAM 2 / SAM 3 segmenter. 
    Refines BBoxes into masks for instance segmentation datasets.
    """
    def __init__(self, model_path: str):
        # Load the Ultralytics SAM, SAM2 or SAM3 model (e.g., "sam3.pt", "sam2_b.pt", "sam_b.pt")
        self.model = SAM(model_path)

    def process_image(self, image_path: str, bboxes: list):
        """
        Produce masks from a list of BBoxes for a single image.
        bboxes: list of [ymin, xmin, ymax, xmax] pixel coordinates.
        """
        # Convert [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax] for Ultralytics
        input_boxes = [[b[1], b[0], b[3], b[2]] for b in bboxes]
        
        # Perform inference with Ultralytics SAM
        results = self.model(image_path, bboxes=input_boxes, verbose=False)
        result = results[0]
        
        # Extract masks and compute dummy scores if not available
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy() # shape: (N, H, W)
            # Ultralytics SAM sometimes doesn't return confidences per mask when prompted. 
            # We'll default to 1.0 or extract if available
            scores = np.ones(len(masks)) 
            return masks, scores
        else:
            return None, None

def rle_encode(mask):
    """Simple RLE encoding for saving masks in JSON efficiently."""
    pixels = np.array(mask, dtype=bool).flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main():
    parser = argparse.ArgumentParser(description="Ultralytics SAM Segmenter")
    parser.add_argument("labels_json", type=str, help="Output from previous tools")
    parser.add_argument("--model", type=str, required=True, help="Ultralytics model (e.g., sam2_b.pt)")
    parser.add_argument("--output", type=str, default="segment_labels.json", help="Final output file")
    
    args = parser.parse_args()

    # Load data
    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segmenter = UltralyticsSAMSegmenter(args.model)
    
    final_output = {}

    for img_name, info in data.items():
        img_path = info["image_path"]
        detections = info["detections"]
        
        if not detections:
            continue
            
        pixel_boxes = [d["pixel_box"] for d in detections]
        print(f"Segmenting {img_name} using Ultralytics ({len(pixel_boxes)} masks)...")
        
        masks, scores = segmenter.process_image(img_path, pixel_boxes)
        
        if masks is not None:
            # Package masks back into JSON
            for i, detection in enumerate(detections):
                mask_bool = masks[i] # (H, W)
                detection["mask_rle"] = rle_encode(mask_bool)
                detection["mask_score"] = float(scores[i])
                
        final_output[img_name] = info

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    
    print(f"Segmented labels saved to {args.output}")

if __name__ == "__main__":
    main()
