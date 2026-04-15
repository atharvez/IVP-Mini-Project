import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

class ClothingDetector:
    """
    Advanced Fashion Detection using a Conditional DETR (Transformer) model.
    Fine-tuned on ModaNet and Fashionpedia.
    """

    def __init__(self, model_name="yainage90/fashion-object-detection"):
        print(f"Loading advanced fashion transformer: {model_name}...")
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        
        # Label mapping for premium descriptors
        self.label_map = {
            "top": "Shirt/T-Shirt",
            "bottom": "Pants/Shorts",
            "outer": "Jacket/Coat",
            "dress": "Dress",
            "hat": "Hat",
            "bag": "Bag",
            "shoes": "Shoes"
        }

    def detect_objects(self, frame, threshold=0.4):
        """
        Detects fashion items using the DETR Transformer.
        """
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        with torch.no_grad():
            inputs = self.image_processor(images=[pil_image], return_tensors="pt")
            outputs = self.model(**inputs.to(self.device))
            
            # Post-process detections
            target_sizes = torch.tensor([[pil_image.size[1], pil_image.size[0]]])
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes
            )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = score.item()
            label_idx = label.item()
            raw_label = self.model.config.id2label[label_idx].lower()
            x1, y1, x2, y2 = map(int, box.tolist())
            
            roi = frame[max(0, y1):min(frame.shape[0], y2), 
                        max(0, x1):min(frame.shape[1], x2)]
            
            detections.append({
                "box": (x1, y1, x2, y2),
                "label": raw_label,
                "confidence": score,
                "roi": roi if roi.size > 0 else None
            })
        
        return detections

    def get_clothing_regions(self, frame):
        """
        Main entry point for unified detection.
        Returns a list of region dicts compatible with the main loop.
        """
        items = self.detect_objects(frame)
        
        regions = []
        for d in items:
            raw_label = d["label"]
            friendly_label = self.label_map.get(raw_label, raw_label.capitalize())
            
            # Determine region type for color palette selection
            region_type = "upper" if raw_label in ["top", "outer", "dress"] else "lower"
            
            regions.append({
                "roi": d["roi"],
                "type": region_type,
                "label": friendly_label,
                "box": d["box"]
            })
        return regions

    def get_edge_map(self, frame):
        """Legacy compatibility."""
        return np.zeros(frame.shape[:2], dtype=np.uint8)
