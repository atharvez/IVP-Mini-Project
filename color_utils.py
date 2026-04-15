import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

class ColorClassifier:
    """
    State-of-the-Art Color Classification using a SigLIP-based vision model
    specifically fine-tuned for fashion products.
    Recognizes 46 professional fashion colors.
    """
    def __init__(self, model_id="prithivMLmods/Fashion-Product-baseColour"):
        print(f"Loading high-fidelity fashion color model: {model_id}...")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(self.device).eval()
        self.labels = self.model.config.id2label

    def _get_central_fabric_patch(self, region):

        h, w = region.shape[:2]
        ch, cw = int(h * 0.25), int(w * 0.25)
        bh, bw = int(h * 0.50), int(w * 0.50)
        
        if bh > 10 and bw > 10:
            return region[ch:ch+bh, cw:cw+bw]
        return region

    def predict_color(self, region):
        if region is None or region.size == 0:
            return "Unknown"

        fabric_patch = self._get_central_fabric_patch(region)
        
        pil_img = Image.fromarray(cv2.cvtColor(fabric_patch, cv2.COLOR_BGR2RGB))
        
        try:
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                idx = outputs.logits.argmax(-1).item()
                color_name = self.labels[idx]
                
                return color_name.replace(" (Base Colour)", "").capitalize()
        except Exception as e:
            print(f"Color Inference Error: {e}")
            return "Unknown"

class ClothingItemClassifier:
    def __init__(self):
        pass
    def predict_item(self, region):
        return "Item"
