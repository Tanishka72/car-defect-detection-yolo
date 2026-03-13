from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# BGR colours for each damage class (for bounding-box drawing)
CLASS_COLORS = {
    "scratch": (0, 0, 255),        # red
    "dent": (0, 165, 255),         # orange
    "crack": (0, 0, 180),          # dark red
    "glass_shatter": (255, 0, 0),  # blue
    "lamp_broken": (0, 255, 255),  # yellow
    "tire_flat": (128, 0, 128),    # purple
    "normal": (0, 255, 0),         # green (no damage)
}
DEFAULT_COLOR = (200, 200, 200)  # gray fallback


@dataclass
class Detection:
    """Single detected defect."""
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


@dataclass
class ImageResult:
    """Detection results for one image."""
    image_path: Path
    detections: list[Detection] = field(default_factory=list)
    error: str | None = None

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


class ImageDetector:
    """Loads a trained YOLOv8 model and runs inference on images."""

    def __init__(self, model_path: str | Path = "models/best.pt", conf: float = 0.25):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.conf = conf
        logger.info("Loaded model: %s (confidence=%.2f)", model_path.name, conf)

    def detect(self, image_path: str | Path) -> ImageResult:
        """Run detection on a single image.

        Skips corrupt or unreadable images gracefully.
        """
        image_path = Path(image_path)
        result = ImageResult(image_path=image_path)

        # Read image with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            result.error = "Corrupt or unreadable image"
            logger.warning("Skipping unreadable image: %s", image_path.name)
            return result

        # Run YOLO inference
        predictions = self.model(img, conf=self.conf, verbose=False)[0]

        for box in predictions.boxes:
            cls_id = int(box.cls[0])
            det = Detection(
                class_name=predictions.names[cls_id],
                confidence=round(float(box.conf[0]), 4),
                bbox=[round(v, 1) for v in box.xyxy[0].tolist()],
            )
            result.detections.append(det)

        logger.info(
            "%s — %d defect(s) found", image_path.name, len(result.detections)
        )
        return result

    def draw_boxes(self, image_path: str | Path, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes on the image and return annotated BGR array."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return img

    def process_and_save(
        self, image_path: str | Path, output_dir: str | Path
    ) -> ImageResult:
        """Detect defects, draw bounding boxes, and save annotated image."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.detect(image_path)

        if result.error:
            return result

        if result.has_detections:
            annotated = self.draw_boxes(image_path, result.detections)
        else:
            annotated = cv2.imread(str(image_path))

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated)
        return result
