import cv2
import numpy as np

def enhance_image(frame):
    """
    Apply image enhancement techniques:
    1. Brightness & Contrast Adjustment
    2. Adaptive Histogram Equalization (CLAHE)
    3. Noise Reduction (Gaussian Blur)
    4. Edge Detection (Sobel Filter - for visualization/analysis)
    5. Color Enhancement (HSV saturation boost)
    """
    
    # 1. Automatic Brightness & Contrast Adjustment
    # Using simple linear scaling: frame_out = alpha * frame + beta
    avg_brightness = np.mean(frame)
    target_brightness = 127
    brightness_diff = target_brightness - avg_brightness
    
    # Dynamic alpha (contrast) and beta (brightness)
    alpha = 1.1 # Slightly increase contrast
    beta = brightness_diff * 0.5 # Adjust brightness halfway to target
    
    enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # 2. Adaptive Histogram Equalization (CLAHE)
    # Convert to LAB to only apply to lightness channel
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. Noise Reduction (Gaussian Blur)
    # Subtle blur to reduce sensor noise without losing too much detail
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 4. Color Enhancement (HSV Saturation Boost)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s = s * 1.2 # Boost saturation by 20%
    s = np.clip(s, 0, 255)
    enhanced = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(enhanced.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    # 5. Edge Detection (Sobel - optional for boundary highlighting)
    # We won't apply this to the 'final' feed directly to keep it realistic,
    # but we can return it or use it for object boundary analysis if needed.
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    
    return enhanced, sobel
