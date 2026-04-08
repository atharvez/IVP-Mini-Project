import cv2
import time
import numpy as np
from enhancement import enhance_image
from detection import ClothingDetector
from color_utils import ColorClassifier
from recommender import FashionRecommender

def main():
    print("Starting Real-Time Outfit Detection and Color Recommendation System...")
    
    # Initialize components
    detector = ClothingDetector(model_name="yolov8n.pt")
    color_clf = ColorClassifier(use_hf=True) # Set to False to skip HF model loading
    recommender = FashionRecommender(model="llama3")
    
    # Setup Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Performance tracking
    prev_time = 0
    
    print("\n--- Controls ---")
    print("Press 'Q' to Exit")
    print("Press 'E' to toggle Enhancement (default: ON)")
    print("----------------\n")
    
    use_enhancement = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Image Enhancement (Phase 1)
        display_frame = frame.copy()
        if use_enhancement:
            display_frame, edges = enhance_image(frame)
            
        # 2. YOLO-Based Detection (Phase 2)
        detections = detector.detect_objects(frame) # Detect on raw frame for consistency
        
        # Draw bounding boxes and analyze regions
        detected_colors = {}
        regions = detector.get_clothing_regions(frame, detections)
        
        for region_data in regions:
            roi = region_data["roi"]
            label = region_data["label"]
            x1, y1, x2, y2 = region_data["parent_box"]
            
            # 3. Color Extraction (Phase 3)
            color_name = color_clf.predict_color(roi)
            detected_colors[label] = color_name
            
            # Draw on display frame
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{label}: {color_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4. LLM Recommendation (Optional Module)
        # We only pass detected colors for recommendation
        recommendation = recommender.generate_recommendation(detected_colors)
        
        # Display recommendation
        # Create a simple overlay at the bottom
        h, w, _ = display_frame.shape
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        cv2.putText(display_frame, recommendation, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Output Display & FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Enhancement: {'ON' if use_enhancement else 'OFF'}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Outfit Detection & Color Recommender", display_frame)
        
        # Input handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            use_enhancement = not use_enhancement
            print(f"Enhancement toggled to: {use_enhancement}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
