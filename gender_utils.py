import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

class GenderDetector:
    """
    Detects gender using a combination of OpenCV Haar Cascades for face
    localization and a Hugging Face Vision Transformer for classification.
    """
    def __init__(self, model_id="prithivMLmods/Gender-Classifier-Mini"):
        print(f"Loading gender classification model: {model_id}...")
        
        # 1. Load Face Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 2. Load HF Model
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.labels = self.model.config.id2label # Usually 0: Female, 1: Male
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()

    def detect_gender(self, frame):
        """
        Identify gender from the largest face found in the frame.
        Returns 'Male', 'Female', or 'Unknown'.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return "Unknown"
        
        # Pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        
        # Crop and predict
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return "Unknown"
            
        try:
            pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                idx = outputs.logits.argmax(-1).item()
                gender = self.labels[idx]
                return gender.capitalize()
        except Exception as e:
            print(f"Gender Detection Error: {e}")
            return "Unknown"
