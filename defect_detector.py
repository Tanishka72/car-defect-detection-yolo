import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import clip


class DefectDetector:
    """Detects scratches, dents, cracks, and defects on objects without custom training.

    Uses:
    - OpenCV image processing to visually highlight anomalies
    - CLIP zero-shot classification to identify defect types
    """

    DEFECT_LABELS = [
        "a clean undamaged surface",
        "a scratch on the surface",
        "a dent or deformation",
        "a crack or fracture",
        "rust or corrosion",
        "a chip or broken piece",
        "discoloration or stain",
        "wear and tear damage",
        "a hole or puncture",
        "peeling or flaking",
    ]

    DEFECT_SHORT = [
        "Clean",
        "Scratch",
        "Dent",
        "Crack",
        "Rust",
        "Chip",
        "Stain",
        "Wear",
        "Hole",
        "Peeling",
    ]

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_features = self._encode_labels()

    def _encode_labels(self):
        """Pre-encode defect label text embeddings."""
        tokens = clip.tokenize(self.DEFECT_LABELS).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def classify_defects(self, image: Image.Image):
        """Use CLIP zero-shot to classify the type of defect in an image region.

        Returns list of (label, probability) sorted by probability descending.
        """
        img_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(img_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()
        results = []
        for i, prob in enumerate(probs):
            results.append((self.DEFECT_SHORT[i], float(prob)))
        results.sort(key=lambda x: -x[1])
        return results

    def detect_surface_anomalies(self, image: Image.Image, sensitivity: int = 50):
        """Use OpenCV edge/contour analysis to find potential scratches and defects.

        Args:
            image: PIL Image to analyze
            sensitivity: 0-100, higher = detect more (including minor) anomalies

        Returns:
            anomaly_mask: PIL Image showing detected anomaly regions
            anomaly_overlay: PIL Image with anomalies highlighted on original
            contour_count: number of anomaly regions found
            anomaly_percentage: percentage of surface area with anomalies
        """
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Adaptive parameters based on sensitivity
        blur_k = max(3, 11 - (sensitivity // 10) * 2)
        if blur_k % 2 == 0:
            blur_k += 1
        canny_low = max(10, 80 - sensitivity)
        canny_high = max(30, 200 - sensitivity * 2)
        min_area = max(5, 100 - sensitivity)

        # Preprocessing: blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        # Multi-scale edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)

        # Morphological operations to connect nearby edges (scratches are linear)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_col = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated_h = cv2.dilate(edges, kernel_line, iterations=1)
        dilated_v = cv2.dilate(edges, kernel_col, iterations=1)
        combined = cv2.bitwise_or(dilated_h, dilated_v)

        # Close small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

        # Find contours (potential defect regions)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # Create anomaly mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, significant_contours, -1, 255, thickness=cv2.FILLED)

        # Create colored overlay
        overlay = img_np.copy()
        # Red highlight for anomalies
        overlay[mask > 0] = [255, 60, 60]
        # Blend with original
        alpha = 0.4
        blended = cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0)

        # Draw contour outlines in bright red
        cv2.drawContours(blended, significant_contours, -1, (255, 0, 0), 2)

        # Stats
        total_pixels = gray.shape[0] * gray.shape[1]
        anomaly_pixels = np.count_nonzero(mask)
        anomaly_pct = (anomaly_pixels / total_pixels) * 100

        anomaly_mask = Image.fromarray(mask)
        anomaly_overlay = Image.fromarray(blended)

        return anomaly_mask, anomaly_overlay, len(significant_contours), anomaly_pct

    def analyze_full(self, image: Image.Image, sensitivity: int = 50):
        """Run full defect analysis: CLIP classification + surface anomaly detection.

        Returns a dict with all results.
        """
        # CLIP zero-shot defect classification
        classifications = self.classify_defects(image)

        # OpenCV anomaly detection
        anomaly_mask, anomaly_overlay, contour_count, anomaly_pct = (
            self.detect_surface_anomalies(image, sensitivity)
        )

        # Determine overall condition
        top_defect = classifications[0]
        if top_defect[0] == "Clean" and top_defect[1] > 0.3:
            condition = "Good"
            condition_color = "green"
        elif anomaly_pct > 5 or top_defect[1] > 0.4 and top_defect[0] != "Clean":
            condition = "Defective"
            condition_color = "red"
        else:
            condition = "Minor Issues"
            condition_color = "orange"

        return {
            "classifications": classifications,
            "anomaly_mask": anomaly_mask,
            "anomaly_overlay": anomaly_overlay,
            "contour_count": contour_count,
            "anomaly_percentage": anomaly_pct,
            "condition": condition,
            "condition_color": condition_color,
        }
