# YOLO Image Classification System

A web-based application for object detection and image classification using YOLOv8.

## Features
- Upload images for instant object detection.
- Visualizes bounding boxes and class labels.
- Shows confidence scores for each detection.
- Fast inference using YOLOv8 nano model.
- Responsive and modern web interface.

## Quick Start
1. Install requirements: `pip install -r requirements.txt`
2. Start the server: `python app.py`
3. Visit: `http://127.0.0.1:5000`

> [!NOTE]
> The system now uses the `yolov8s.pt` model, capable of detecting 80 classes including humans, vehicles, and animals.

## Tech Stack
- **Backend**: Python, Flask
- **Model**: YOLOv8 (Ultralytics)
- **Frontend**: HTML5, CSS3 (Vanilla)
