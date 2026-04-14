import cv2
import numpy as np


def enhance_image(frame):
    """
    Apply spatial-domain image enhancement:
      1. Brightness & Contrast auto-adjustment
      2. Adaptive Histogram Equalization (CLAHE) on L-channel
      3. Gaussian noise reduction
      4. HSV saturation boost
    Returns (enhanced_frame, sobel_edge_map).
    """

    # 1. Automatic Brightness & Contrast Adjustment
    avg_brightness = np.mean(frame)
    target_brightness = 127
    brightness_diff = target_brightness - avg_brightness
    alpha = 1.1
    beta  = brightness_diff * 0.5
    enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # 2. CLAHE on LAB lightness channel
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # 3. Gaussian noise reduction
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 4. HSV saturation boost (+20 %)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.2, 0, 255)
    enhanced = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(enhanced.astype("uint8"), cv2.COLOR_HSV2BGR)

    # 5. Sobel edge map (returned for optional overlay/analysis use)
    gray   = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel  = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))

    return enhanced, sobel


def get_clothing_edges(frame, canny_low=30, canny_high=100):
    """
    Dedicated Canny-based edge extractor tuned for clothing boundaries.
    Used by ClothingDetector internally; exposed here for IVP analysis.

    Steps:
      1. Convert to grayscale
      2. CLAHE to normalise lighting
      3. Gaussian blur to reduce textile noise
      4. Canny edge detection
      5. Morphological close to fill clothing contour gaps

    Returns:
      edges      – raw Canny edge map (uint8)
      closed     – morphologically closed edge map (better for contours)
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray    = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges   = cv2.Canny(blurred, canny_low, canny_high)

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return edges, closed
