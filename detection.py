import cv2
import numpy as np
from ultralytics import YOLO


class ClothingDetector:
    """
    Detects clothing using a two-stage pipeline:
      Stage 1 – YOLOv8 finds the *person* region (used only as a crop hint).
      Stage 2 – Canny edge detection + contour analysis isolates actual
                 clothing boundaries within that person crop.
    No bounding rectangles are drawn; callers receive contour polygons.
    """

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

        # Only "Person" is used as an ROI hint from YOLO
        self.person_class_id = 0

        # Canny thresholds (tuned for clothing texture)
        self.canny_low = 30
        self.canny_high = 100

        # Minimum contour area to be considered a clothing region (px²)
        self.min_contour_area = 3000

    # ------------------------------------------------------------------
    # Stage 1: Locate person crops via YOLO
    # ------------------------------------------------------------------
    def _get_person_boxes(self, frame):
        """Return list of (x1,y1,x2,y2) for each detected person."""
        results = self.model(frame, verbose=False)[0]
        boxes = []
        h, w = frame.shape[:2]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == self.person_class_id and conf > 0.40:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                boxes.append((x1, y1, x2, y2))
        return boxes

    # ------------------------------------------------------------------
    # Stage 2: Edge + contour extraction inside a person crop
    # ------------------------------------------------------------------
    def _extract_clothing_contours(self, crop, offset_x, offset_y):
        """
        Run Canny edge detection on a person crop and return clothing contours
        in *frame* coordinate space, split into upper/lower halves.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE to survive varying lighting conditions
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Slight Gaussian blur to suppress noise before Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # Canny edge map
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Morphological close to bridge small gaps in clothing edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find external contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_crop, w_crop = crop.shape[:2]
        mid_y = int(h_crop * 0.45)   # upper/lower clothing split

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            # Shift contour back to full-frame coordinates
            shifted = cnt + np.array([[[offset_x, offset_y]]])

            # Determine centre of mass y-coord (crop space) for upper/lower tag
            M = cv2.moments(cnt)
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            region_type = "upper" if cy < mid_y else "lower"

            # Tight bounding rect (used for ROI crop only – never drawn)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            roi_x1 = max(0, bx + offset_x)
            roi_y1 = max(0, by + offset_y)
            roi_x2 = roi_x1 + bw
            roi_y2 = roi_y1 + bh

            regions.append({
                "contour": shifted,           # polygon in frame coords
                "type": region_type,
                "roi_coords": (roi_x1, roi_y1, roi_x2, roi_y2),
            })

        return regions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_objects(self, frame):
        """Kept for compatibility – returns person detections."""
        return self._get_person_boxes(frame)

    def get_clothing_regions(self, frame, person_boxes):
        """
        Main entry point.
        Returns list of dicts:
          {
            'contour':    np.ndarray  – shape polygon (frame coords),
            'type':       str         – 'upper' | 'lower',
            'roi':        np.ndarray  – BGR crop of the clothing region,
            'roi_coords': tuple       – (x1,y1,x2,y2) of the ROI (never drawn),
          }
        """
        all_regions = []

        if not person_boxes:
            # Fallback: run edge detection on full frame
            person_boxes = [(0, 0, frame.shape[1], frame.shape[0])]

        for (x1, y1, x2, y2) in person_boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            regions = self._extract_clothing_contours(crop, x1, y1)
            for r in regions:
                rx1, ry1, rx2, ry2 = r["roi_coords"]
                roi = frame[ry1:ry2, rx1:rx2]
                r["roi"] = roi if roi.size > 0 else crop
                all_regions.append(r)

        return all_regions

    def get_edge_map(self, frame):
        """
        Return a full-frame Canny edge map (for visualisation / overlay use).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges
