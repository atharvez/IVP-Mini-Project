from ultralytics import YOLO
import cv2

class ClothingDetector:
    def __init__(self, model_name="yolov8n.pt"):
        # load YOLOv8 model - downloads automatically if not present
        self.model = YOLO(model_name)
        # Class names for YOLOv8 (COCO dataset)
        # 0: person, 24: backpack, 26: handbag, 27: tie, 28: suitcase, 56: chair...
        # Specific clothing-related IDs:
        # shirt (not in coco directly, usually detected as 'person' or part of it)
        # COCO actually has: person (0), tie (27), handbag (26)
        # However, many YOLOv8 models can detect specific items if they are specialized.
        # For the standard model, we often filter for person OR use a specialized model.
        # To keep it generic and functional for a demo, we will use the standard COCO 
        # classes but we will also handle generic detections.
        
        # Mapping for display labels
        self.clothing_classes = {
            0: "Person",
            24: "Backpack",
            26: "Handbag",
            27: "Tie",
            28: "Suitcase",
            # We can extend this list
        }

    def detect_objects(self, frame):
        """
        Detect objects in frame using YOLOv8.
        Returns detections with bounding boxes.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # We filter for confidence and classes of interest
            if conf > 0.4:
                # For this demo, we'll return all relevant detections
                # If it's a 'person', we'll try to analyze the upper/lower halves for clothes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                label = self.clothing_classes.get(cls_id, f"Object {cls_id}")
                
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "label": label,
                    "confidence": conf,
                    "class_id": cls_id
                })
        
        return detections

    def get_clothing_regions(self, frame, detections):
        """
        Crop regions from the frame based on detections.
        Special handling: If person is detected, crop upper and lower halves.
        """
        regions = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            roi = frame[y1:y2, x1:x2]
            
            if det["label"] == "Person":
                # Split person into approximate Upper and Lower regions
                h = y2 - y1
                mid = y1 + int(h * 0.4) # Approx split for shirt/pants
                
                upper_roi = frame[y1:mid, x1:x2]
                lower_roi = frame[mid:y2, x1:x2]
                
                if upper_roi.size > 0:
                    regions.append({"roi": upper_roi, "label": "Upper Wear", "parent_box": (x1, y1, x2, mid)})
                if lower_roi.size > 0:
                    regions.append({"roi": lower_roi, "label": "Lower Wear", "parent_box": (x1, mid, x2, y2)})
            else:
                if roi.size > 0:
                    regions.append({"roi": roi, "label": det["label"], "parent_box": (x1, y1, x2, y2)})
                    
        return regions
