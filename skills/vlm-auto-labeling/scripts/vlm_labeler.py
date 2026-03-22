import os
import json
import argparse
import base64
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from typing import List, Dict, Any

# Configure VLM Labeler
# Usage: python vlm_labeler.py data/images/ --prompt "detect cats" --api_key YOUR_API_KEY --output labels.json

class VLMLabeler:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def _encode_image(self, image_path: Path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def detect_objects(self, image_path: Path, prompt: str) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using VLM and return structured JSON.
        """
        img = Image.open(image_path)
        width, height = img.size
        
        # System instruction to enforce JSON format
        system_prompt = f"""
        {prompt}
        Identify objects and return bounding boxes in [ymin, xmin, ymax, xmax] format normalized (0-1000).
        Return ONLY valid JSON format like:
        [
          {{"label": "cat", "box": [100, 200, 500, 600]}},
          {{"label": "dog", "box": [50, 400, 200, 800]}}
        ]
        Do not include markdown tags or explanations.
        """
        
        response = self.model.generate_content([system_prompt, img])
        
        try:
            # Simple cleanup for JSON extraction
            content = response.text.strip()
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            elif "```" in content:
                 content = content.split("```")[-1].split("```")[0].strip()
            
            val = json.loads(content)
            
            # Map normalized to pixel coords (optional, or keep normalized for SAM)
            for item in val:
                ymin, xmin, ymax, xmax = item["box"]
                item["pixel_box"] = [
                    int(ymin * height / 1000),
                    int(xmin * width / 1000),
                    int(ymax * height / 1000),
                    int(xmax * width / 1000)
                ]
            return val
        except Exception as e:
            print(f"Error parsing VLM response for {image_path.name}: {e}")
            print(f"Raw Response: {response.text}")
            return []

def main():
    parser = argparse.ArgumentParser(description="VLM Auto-Labeler")
    parser.add_argument("image_dir", type=str, help="Directory containing images")
    parser.add_argument("--prompt", type=str, required=True, help="Description of objects to label")
    parser.add_argument("--api_key", type=str, help="API Key for VLM (default from GEMINI_API_KEY env)")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="VLM Model name")
    parser.add_argument("--output", type=str, default="auto_labels.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: API Key not provided. Use --api_key or GEMINI_API_KEY env.")
        return

    labeler = VLMLabeler(api_key, args.model)
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    results = {}
    
    print(f"Starting auto-labeling for {len(image_paths)} images...")
    for idx, img_path in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] Processing {img_path.name}...")
        detections = labeler.detect_objects(img_path, args.prompt)
        results[img_path.name] = {
            "image_path": str(img_path),
            "detections": detections
        }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_all_unicode=False)
    
    print(f"Finished! Results saved to {args.output}")

if __name__ == "__main__":
    main()
