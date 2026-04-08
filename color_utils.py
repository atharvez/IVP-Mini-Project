import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

class ColorClassifier:
    def __init__(self, use_hf=True):
        self.use_hf = use_hf
        if self.use_hf:
            try:
                print("Loading Hugging Face color model (prithivMLmods/Fashion-Product-baseColour)...")
                self.model_name = "prithivMLmods/Fashion-Product-baseColour"
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
                # Load labels
                self.labels = self.model.config.id2label
                print("Hugging Face model loaded successfully.")
            except Exception as e:
                print(f"Error loading HF model: {e}. Falling back to HSV logic.")
                self.use_hf = False

    def get_color_hsv(self, region):
        """
        Extract dominant color using Mean HSV as a fallback.
        """
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Mask out very dark or very bright pixels (background/shadows)
        mask = cv2.inRange(hsv_region, (0, 20, 20), (180, 255, 255))
        mean_hsv = cv2.mean(hsv_region, mask=mask)[:3]
        return mean_hsv

    def get_color_name_hsv(self, hsv):
        """
        Map HSV values to basic color names.
        """
        h, s, v = hsv
        if v < 40: return "Black"
        if v > 200 and s < 40: return "White"
        if s < 40: return "Gray"
        
        if h < 10 or h > 165: return "Red"
        if h < 25: return "Orange"
        if h < 35: return "Yellow"
        if h < 85: return "Green"
        if h < 130: return "Blue"
        if h < 165: return "Purple"
        
        return "Unknown"

    def predict_color(self, region):
        """
        Predict color of the region.
        """
        if self.use_hf:
            try:
                # Convert OpenCV BGR to PIL Image
                image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                inputs = self.processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                predicted_class_idx = logits.argmax(-1).item()
                return self.labels[predicted_class_idx]
            except Exception as e:
                print(f"HF Inference Error: {e}")
                # Fallback to HSV
                hsv = self.get_color_hsv(region)
                return self.get_color_name_hsv(hsv)
        else:
            hsv = self.get_color_hsv(region)
            return self.get_color_name_hsv(hsv)
