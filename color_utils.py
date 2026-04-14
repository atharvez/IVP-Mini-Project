import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

class ClothingItemClassifier:
    # Ranked list of verified HuggingFace model IDs (tried in order)
    _MODEL_CANDIDATES = [
        "arize-ai/resnet-50-fashion-mnist-quality-drift",  # ResNet50, FashionMNIST (T-Shirt/Trouser/Dress…)
        "tzhao3/vit-FashionMNIST",                         # ViT fine-tuned on FashionMNIST
        "Methmani/ImageClassification_fashion-mnist",       # ResNet, FashionMNIST
    ]

    def __init__(self):
        self.model = None
        self.processor = None
        self.labels = {}
        for model_id in self._MODEL_CANDIDATES:
            try:
                print(f"Loading clothing model ({model_id})...")
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model    = AutoModelForImageClassification.from_pretrained(model_id)
                self.labels   = self.model.config.id2label
                print(f"Clothing model loaded: {model_id}")
                break
            except Exception as e:
                print(f"  Could not load {model_id}: {e}")
        if self.model is None:
            print("No remote clothing model available – using heuristic fallback.")

    def predict_item(self, region):
        """
        Predict the type of clothing item (Shirt, T-Shirt, Pants, etc.)
        """
        if self.model is None or region is None or region.size == 0:
            return "Item"
        
        try:
            # Convert OpenCV BGR to PIL Image
            image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            idx = logits.argmax(-1).item()
            raw_label = self.labels[idx].lower()
            
            # Map raw labels to user-friendly names
            mapping = {
                "t-shirt/top": "T-Shirt",
                "trouser": "Pants",
                "pullover": "Sweater",
                "dress": "Dress",
                "coat": "Coat",
                "shirt": "Shirt",
                "suit": "Suit",
                "skirt": "Skirt"
            }
            
            for key, val in mapping.items():
                if key in raw_label:
                    return val
            return raw_label.capitalize()
            
        except Exception as e:
            print(f"Clothing Classification Error: {e}")
            return "Clothing"

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
                print("Hugging Face color model loaded successfully.")
            except Exception as e:
                print(f"Error loading color model: {e}. Falling back to HSV logic.")
                self.use_hf = False

    def get_color_hsv(self, region):
        """
        Extract dominant color using Mean HSV as a fallback.
        """
        if region.size == 0: return (0, 0, 0)
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
        if region.size == 0: return "Unknown"
        
        if self.use_hf:
            try:
                # Convert OpenCV BGR to PIL Image
                image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                inputs = self.processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                predicted_class_idx = logits.argmax(-1).item()
                return self.labels[predicted_class_idx].capitalize()
            except Exception as e:
                print(f"Color Inference Error: {e}")
                # Fallback to HSV
                hsv = self.get_color_hsv(region)
                return self.get_color_name_hsv(hsv)
        else:
            hsv = self.get_color_hsv(region)
            return self.get_color_name_hsv(hsv)
