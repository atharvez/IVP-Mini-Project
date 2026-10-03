# IVP Mini Project -- AI Fashion and Person Analysis

Image and Video Processing mini project using YOLOv8 for person detection, gender classification, and fashion recommendation.

## Overview

A computer vision pipeline that:

1. Detects persons using YOLOv8
2. Classifies gender from detected regions
3. Analyzes clothing colors for style matching
4. Recommends fashion items based on detected styles

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Language | Python 3.9+ |

## Project Structure

```
IVP-Mini-Project/
|-- main.py                 # Pipeline entry point
|-- detection.py            # YOLOv8 person detection
|-- enhancement.py          # Image pre-processing
|-- color_utils.py          # Clothing color extraction
|-- gender_utils.py         # Gender classification
|-- recommender.py          # Fashion recommendation engine
|-- train_fashion.py        # Fashion classifier training
|-- yolov8n.pt             # Pre-trained YOLOv8 weights
\-- requirements.txt
```

## Getting Started

```bash
git clone https://github.com/atharvez/IVP-Mini-Project.git
cd IVP-Mini-Project
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py --image Images/sample.jpg
```

## Course

Image and Video Processing (IVP) Mini Project -- Atharva Desai
