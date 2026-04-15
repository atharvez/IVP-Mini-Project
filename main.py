import cv2
import time
import numpy as np
import argparse
import os
from enhancement import enhance_image
from detection import ClothingDetector
from color_utils import ColorClassifier, ClothingItemClassifier
from recommender import FashionRecommender
from gender_utils import GenderDetector

# ── Colour palette for premium UI ──────────────────────────────────────────
UI_COLORS = {
    "upper":  (255, 120, 0),    # deep orange
    "lower":  (0, 200, 255),    # cyan
    "accent": (180, 0, 255),    # purple
}

def draw_box_label(frame, box, label, color):
    """Draw a stylish bounding box with a high-readability label pill."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    label_y = max(y1 - 10, 25)
    cv2.rectangle(frame, (x1, label_y - th - 10), (x1 + tw + 15, label_y + 5), color, -1)
    cv2.putText(frame, label, (x1 + 8, label_y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def process_single_frame(frame, detector, color_clf, gender_clf, recommender, 
                         active_gender="Unknown", use_enhancement=True, detect_gender=True):
    """
    Unified AI pipeline: Enhancement -> Detection -> Color -> Gender -> Recommendation.
    """
    # ── Step 1: Image Enhancement
    display_frame = frame.copy()
    if use_enhancement:
        display_frame, _ = enhance_image(frame)

    # ── Step 2: Gender Detection
    if detect_gender:
        active_gender = gender_clf.detect_gender(frame)

    # ── Step 3: Clothing Detection
    regions = detector.get_clothing_regions(frame)

    # ── Step 4: Classification & UI
    outfit_data = {"upper": None, "lower": None}
    for region in regions:
        roi, box, r_type, raw_label = region["roi"], region["box"], region["type"], region["label"]
        if roi is None or roi.size == 0: continue
        
        color_name = color_clf.predict_color(roi)
        if r_type in outfit_data and outfit_data[r_type] is None:
            outfit_data[r_type] = {"color": color_name, "label": raw_label}

        ui_color = UI_COLORS.get(r_type, (200, 200, 200))
        draw_box_label(display_frame, box, f"{raw_label}: {color_name}", ui_color)

    # ── Step 5: Recommendation
    recommendation = recommender.generate_recommendation(outfit_data, gender=active_gender)
    h, w, _ = display_frame.shape
    
    # Draw Recommendation Banner
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, h - 85), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)
    cv2.putText(display_frame, f"STYLE SUGGESTION ({active_gender.upper()})", (20, h - 60), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, UI_COLORS["accent"], 1, cv2.LINE_AA)
    
    display_rec = recommendation[:110] if recommendation else "Analyzing outfit..."
    cv2.putText(display_frame, display_rec, (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    return display_frame, active_gender, len(regions)

def main():
    parser = argparse.ArgumentParser(description="Premium Fashion Intelligence System")
    parser.add_argument("--image", "-i", type=str, help="Path to input image for static analysis")
    parser.add_argument("--output", "-o", type=str, help="Path to save the output image")
    args = parser.parse_args()

    print("Starting Premium Outfit AI Detection System…")
    detector, color_clf, gender_clf = ClothingDetector(), ColorClassifier(), GenderDetector()
    recommender = FashionRecommender(model="llama3")

    # ── Mode A: Static Image ────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not load image at {args.image}")
            return
        
        result_frame, gender, items_count = process_single_frame(
            frame, detector, color_clf, gender_clf, recommender, detect_gender=True
        )
        
        print(f"Results: {gender} identified with {items_count} items.")
        if args.output:
            cv2.imwrite(args.output, result_frame)
            print(f"Saved result to {args.output}")
            
        cv2.imshow("Premium Fashion Intelligence (Static Image Mode)", result_frame)
        print("\nPress any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Mode B: Webcam ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time, frame_count = 0, 0
    use_enhancement, active_gender = True, "Unknown"
    
    print("\n--- Controls ---")
    print("  Q  – Quit | E  – Toggle Video Enhancement\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # Periodic Gender detection (every 30 frames)
        detect_gen_this_frame = (frame_count % 30 == 0)
        
        display_frame, active_gender, items_count = process_single_frame(
            frame, detector, color_clf, gender_clf, recommender, 
            active_gender=active_gender, use_enhancement=use_enhancement, 
            detect_gender=detect_gen_this_frame
        )

        # HUD
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.rectangle(display_frame, (10, 10), (220, 120), (30, 30, 30), -1)
        cv2.rectangle(display_frame, (10, 10), (220, 120), (100, 100, 100), 1)
        cv2.putText(display_frame, f"FPS: {int(fps)}", (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"GENDER: {active_gender}", (25, 68), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 100), 1)
        cv2.putText(display_frame, f"ITEMS: {items_count}", (25, 96), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Premium Fashion Intelligence System (Webcam Mode)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('e'): use_enhancement = not use_enhancement

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
