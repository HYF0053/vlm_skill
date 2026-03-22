---
name: "vlm-auto-labeling"
description: "VLM-assisted auto-labeling skill for computer vision datasets. Covers zero-shot detection with Grounding DINO, interactive segmentation with SAM (Segment Anything Model), and VLM-based semantic tagging. Includes workflows for generating COCO/YOLO annotations from raw images using advanced foundation models (VLMs, OWL-ViT, SAM) and human-in-the-loop refinement."
---

# VLM-Assisted Auto-Labeling

Professional data engineering skill for building automated annotation pipelines using Vision Language Models (VLMs) and Visual Foundation Models.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Expertise](#core-expertise)
- [Tech Stack](#tech-stack)
- [Workflow 1: VLM-Guided Zero-Shot Detection](#workflow-1-vlm-guided-zero-shot-detection)
- [Workflow 2: Precision Masking with SAM](#workflow-2-precision-masking-with-sam)
- [Workflow 3: Quality Control & Human-in-the-loop](#workflow-3-quality-control--human-in-the-loop)
- [Common Commands](#common-commands)

## Quick Start

```bash
# 1. Generate Bounding Boxes using Gemini/GPT-4o or Grounding DINO
python scripts/vlm_labeler.py images/ --prompt "detect all defects on the car" --output bboxes.json

# 2. Refine BBoxes into precise Instance Segmentation masks using SAM
python scripts/sam_segmenter.py images/ --bboxes bboxes.json --output masks.json

# 3. Convert results to YOLO or COCO format for training
python scripts/label_converter.py results/ --format yolo --output datasets/
```

## Core Expertise

This skill provides guidance on:

- **Zero-Shot Object Detection**: Grounding DINO, OWL-ViT, Detic
- **Open-Vocabulary Segmentation**: SAM (Segment Anything), SAM 2, FastSAM
- **Semantic Text Coaching**: Using VLMs (Gemini, GPT-4o, LLaVA) for complex visual reasoning to identify corner cases.
- **Auto-Captioning**: BLIP-2, Llava for generating training captions (for Diffusion/CLIP training).
- **Data Distillation**: Converting massive unlabelled data into silver-standard labels for student model training (e.g., YOLO training).

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Foundation Models | SAM & SAM 2, Grounding DINO, OWL-ViT |
| Multimodal APIs | Google Gemini Pro Vision, OpenAI GPT-4o, Groq Llama 3 Vision |
| Libraries | autodistill, supervision, roboflow, segment-anything-2 |
| Annotation Tools | Label Studio, CVAT (with AI integration) |
| Output Formats | COCO, YOLOv8/v11, Pascal VOC |

---

## Workflow 1: VLM-Guided Zero-Shot Detection

Use this workflow to generate initial detection labels without manual annotation.

### Step 1: Initialize Discovery Prompt

Define the visual discovery task for the VLM.

```python
PROMPT = """
Identify all instances of 'cracks' and 'scratches' on the mechanical parts.
Return the coordinates in [y1, x1, y2, x2] normalized format (0-1000).
Format: JSON list of objects with 'label' and 'box'.
"""
```

### Step 2: Batch Generation

```bash
# Run the labeler script with a multi-modal API
python scripts/vlm_labeler.py data/raw/ \
    --vlm gemini-1.5-flash \
    --prompt "detect individual people and their helmets" \
    --confidence 0.6 \
    --output silver_labels.json
```

---

## Workflow 2: Precision Masking with SAM

Convert rough bounding boxes or points into pixel-perfect masks.

### Step 1: Load BBoxes and Run SAM

```bash
# Using SAM 2 for high-speed batch masking
python scripts/sam_segmenter.py data/raw/ \
    --input-labels silver_labels.json \
    --model-size sam2_hct_large \
    --output segment_labels.json
```

### Step 2: Multi-Sample Verification

Calculate the IoU between VLM-predicted BBox and SAM-generated BBox to filter out low-quality auto-labels.

---

## Workflow 3: Quality Control & Human-in-the-loop

Automated labels often require a "silver-to-gold" refinement step.

### Step 1: Consistency Check (Self-Correction)

Ask the VLM to verify its own output after cropping the detected region.

```bash
python scripts/vlm_verifier.py data/raw/ \
    --labels silver_labels.json \
    --check "Does this crop really contain a crack? [Yes/No]"
```

### Step 2: Export to Review Tool

```bash
# Export to Label Studio for final human verification
python scripts/label_converter.py data/ \
    --to label_studio \
    --output review_project/
```

---

## Common Commands

| Command | Description |
|---------|-------------|
| `python scripts/vlm_labeler.py` | Generate initial tags and BBoxes using a VLM. |
| `python scripts/label_converter.py` | Convert JSON labels to COCO/YOLO/VOC format. |
| `python scripts/sam_segmenter.py` | Generate instance masks from BBoxes. |
| `python scripts/dataset_cleaner.py` | Filter out low-confidence auto-labels. |

---

## Resources

- **Foundation Models**: [Segment Anything (SAM) 2](https://github.com/facebookresearch/segment-anything-2)
- **Frameworks**: [Supervision](https://github.com/roboflow/supervision) - Essential for label manipulation.
- **Reference**: [AutoDistill](https://github.com/roboflow/autodistill) - Ecosystem for distilling knowledge from large models into small ones.
