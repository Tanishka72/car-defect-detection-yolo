from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2


class ImageAnalyzer:
    """Detects objects in images using YOLOv8."""

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def analyze(self, image: Image.Image, confidence: float = 0.25):
        """Run object detection on a PIL Image.

        Returns:
            annotated_image: PIL Image with bounding boxes drawn
            detections: list of dicts with class name, confidence, and bbox
        """
        results = self.model(image, conf=confidence)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": results.names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
            })

        # Draw annotated image
        annotated = results.plot()  # returns BGR numpy array
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(annotated_rgb)

        return annotated_image, detections

    def crop_detections(self, image: Image.Image, detections: list):
        """Crop each detected object from the image for per-object defect analysis."""
        crops = []
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            crop = image.crop((x1, y1, x2, y2))
            crops.append({"class": det["class"], "crop": crop, "bbox": det["bbox"]})
        return crops
