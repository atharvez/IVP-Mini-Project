import cv2
import time
import numpy as np
import torch
from enhancement import enhance_image
from detection import ClothingDetector
from color_utils import ColorClassifier, ClothingItemClassifier
from recommender import FashionRecommender

# ── Colour palette for upper / lower contours ──────────────────────────────
CONTOUR_COLORS = {
    "upper":  (0, 200, 255),   # cyan
    "lower":  (180, 0, 255),   # purple
    "object": (0, 255, 120),   # green
}

def draw_contour_label(frame, contour, label, color):
    """
    Draw a filled, semi-transparent contour outline and a floating label.
    No bounding box is ever drawn.
    """
    # Semi-transparent fill
    overlay = frame.copy()
    cv2.drawContours(overlay, [contour], -1, color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # Thick contour outline
    cv2.drawContours(frame, [contour], -1, color, 2)

    # Label positioned above the topmost point of the contour
    top_pt = tuple(contour[contour[:, :, 1].argmin()][0])
    label_y = max(top_pt[1] - 12, 20)

    # Background pill for readability
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame,
                  (top_pt[0] - 4, label_y - th - 6),
                  (top_pt[0] + tw + 4, label_y + 4),
                  color, -1)
    cv2.putText(frame, label, (top_pt[0], label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def blend_edge_overlay(frame, edges, alpha=0.30):
    """
    Blend a coloured edge map on top of the frame for IVP visualisation.
    Edges are tinted cyan so they're distinct from the feed.
    """
    edge_colored = np.zeros_like(frame)
    edge_colored[edges > 0] = (255, 230, 0)   # yellow edge tint
    cv2.addWeighted(edge_colored, alpha, frame, 1.0, 0, frame)


def main():
    print("Starting Edge-Based Outfit Detection System…")
    print("Dataset: DeepFashion / Fashion-MNIST labels via HuggingFace classifiers")

    # ── Init components ──────────────────────────────────────────────────────
    detector   = ClothingDetector(model_name="yolov8n.pt")
    color_clf  = ColorClassifier(use_hf=True)
    item_clf   = ClothingItemClassifier()
    recommender = FashionRecommender(model="llama3")

    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ── State flags ──────────────────────────────────────────────────────────
    prev_time       = 0
    use_enhancement = True
    show_edges      = True      # toggle edge overlay

    print("\n--- Controls ---")
    print("  Q  – Quit")
    print("  E  – Toggle image enhancement")
    print("  V  – Toggle edge map overlay")
    print("----------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Enhancement ──────────────────────────────────────────────────────
        display_frame = frame.copy()
        if use_enhancement:
            display_frame, _ = enhance_image(frame)

        # ── Stage-1: YOLO person crops ───────────────────────────────────────
        person_boxes = detector.detect_objects(frame)

        # ── Stage-2: Edge / contour-based clothing regions ───────────────────
        regions = detector.get_clothing_regions(frame, person_boxes)

        # ── Stage-3: Classify each region ────────────────────────────────────
        detected_colors = {}

        for region in regions:
            roi          = region["roi"]
            contour      = region["contour"]
            region_type  = region["type"]

            if roi is None or roi.size == 0:
                continue

            # --- Item classification (HuggingFace DeepFashion / FashionMNIST) ---
            raw_item = item_clf.predict_item(roi)

            # Fallback heuristic when classifier is uncertain
            if raw_item in ("Item", "Clothing"):
                raw_item = "Shirt/T-Shirt" if region_type == "upper" else "Pants"

            # --- Color classification ---
            color_name = color_clf.predict_color(roi)
            detected_colors[raw_item] = color_name

            # --- Draw contour + label (NO bounding box) ---
            contour_color = CONTOUR_COLORS.get(region_type, (200, 200, 200))
            label_text    = f"{raw_item}: {color_name}"
            draw_contour_label(display_frame, contour, label_text, contour_color)

        # ── Edge overlay (IVP visualisation) ─────────────────────────────────
        if show_edges:
            edge_map = detector.get_edge_map(display_frame)
            blend_edge_overlay(display_frame, edge_map, alpha=0.20)

        # ── Recommendation panel ─────────────────────────────────────────────
        recommendation = recommender.generate_recommendation(detected_colors)
        h, w, _ = display_frame.shape
        overlay  = display_frame.copy()
        cv2.rectangle(overlay, (0, h - 75), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.70, display_frame, 0.30, 0, display_frame)

        cv2.putText(display_frame, recommendation[:100],
                    (20, h - 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
        if len(recommendation) > 100:
            cv2.putText(display_frame, recommendation[100:210],
                        (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (200, 200, 200), 1, cv2.LINE_AA)

        # ── HUD ──────────────────────────────────────────────────────────────
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(display_frame, f"FPS: {int(fps)}",
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Enhance: {'ON' if use_enhancement else 'OFF'}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
        cv2.putText(display_frame, f"Edges: {'ON' if show_edges else 'OFF'}",
                    (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2)
        cv2.putText(display_frame,
                    f"Regions: {len(regions)}",
                    (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 100), 2)

        cv2.putText(display_frame,
                    "Detection: Canny Edges + Contours | Dataset: DeepFashion via HuggingFace",
                    (10, h - 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (160, 160, 255), 1, cv2.LINE_AA)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Edge-Based Clothing Identification System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            use_enhancement = not use_enhancement
        elif key == ord('v'):
            show_edges = not show_edges

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
