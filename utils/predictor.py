import os
import cv2
from ultralytics import YOLO

class YOLOPredictor:
    def __init__(self, model_path='yolov8s.pt'):
        # Load the pre-trained YOLOv8 small model (COCO dataset)
        # This model detects 80 classes including humans, cars, etc.
        self.model = YOLO(model_path)

    def predict(self, image_path, output_path):
        """
        Runs YOLOv8 detection on an image and saves the result.
        Returns a list of detected objects with their labels and confidence scores.
        """
        results = self.model(image_path)
        
        # results is a list, we take the first item
        result = results[0]
        
        # Save the annotated image
        annotated_img = result.plot()
        cv2.imwrite(output_path, annotated_img)
        
        detections = []
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections.append({
                'label': label,
                'confidence': f"{confidence * 100:.2f}%"
            })
            
        return detections
