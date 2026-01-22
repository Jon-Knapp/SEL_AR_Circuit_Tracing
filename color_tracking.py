'''
Minimum Requirements:
opencv-python==4.6.0.66
numpy==1.23.4

Sources:
https://www.youtube.com/watch?v=aFNDh5k3SjU
https://github.com/computervisioneng/color-detection-opencv

Description: This program will make a mask based on a provided BGR value.

In the program below, I chose bright yellow. You may want to choose another color.
If so, make sure to change the "TARGET_BGR" values. Remember, the variable is expecting
a BGR tuple, not a normal RGB tuple.

The aforementioned mask will then be used to create a tracking frame around the
object of that color. If multiple objects of the same color are in frame,
the mask will see them as one object and will create a tracking frame around both objects.

Next steps:
1) Edit the code to have it track multiple colors at the same time.

2) Apply brightly colored tape (colors that aren't used in SEL terminal blocks)
to the tips of multimeter probes and have the program track the tips of the probes.

3) Optional: Have the frames produce a label in the video feed to show "right probe"
and "left probe".

Disclaimers:
This code was produced using the above-mentioned sources and Chat-GPT.
'''

import cv2 as cv
import numpy as np


def get_hsv_ranges_for_bgr(color_bgr, h_tol=10, s_min=100, v_min=100):
    """
    Convert a target BGR color into one (or two) HSV threshold ranges.

    Why “one or two” ranges?
    - Hue in OpenCV HSV is 0..179 and WRAPS around at the ends.
    - Red-ish colors sit near 0 and near 179.
    - If the tolerance window crosses an edge, we split into two ranges.

    Returns:
        List of (lower, upper) np.array pairs, each shape (3,), dtype uint8.
    """
    # Make a 1x1 BGR “image” so OpenCV can convert it to HSV
    pixel = np.uint8([[color_bgr]])
    hsv_pixel = cv.cvtColor(pixel, cv.COLOR_BGR2HSV)[0, 0]
    hue = int(hsv_pixel[0])

    h_low = hue - h_tol
    h_high = hue + h_tol

    ranges = []

    # Case 1: hue window goes below 0 (wraps around)
    if h_low < 0:
        # Range A: [0 .. h_high]
        ranges.append((
            np.array([0, s_min, v_min], dtype=np.uint8),
            np.array([h_high, 255, 255], dtype=np.uint8),
        ))
        # Range B: [(180 + h_low) .. 179]
        ranges.append((
            np.array([180 + h_low, s_min, v_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        ))

    # Case 2: hue window goes above 179 (wraps around)
    elif h_high > 179:
        # Range A: [h_low .. 179]
        ranges.append((
            np.array([h_low, s_min, v_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        ))
        # Range B: [0 .. (h_high - 180)]
        ranges.append((
            np.array([0, s_min, v_min], dtype=np.uint8),
            np.array([h_high - 180, 255, 255], dtype=np.uint8),
        ))

    # Case 3: normal window (no wrap)
    else:
        ranges.append((
            np.array([h_low, s_min, v_min], dtype=np.uint8),
            np.array([h_high, 255, 255], dtype=np.uint8),
        ))

    return ranges


def build_color_mask(hsv_frame, hsv_ranges):
    """
    Given an HSV frame and a list of HSV ranges, build one combined mask.

    Output mask is single-channel uint8 where:
    - 255 means “pixel is inside one of the ranges”
    - 0 means “pixel is outside”
    """
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    # OR together masks for each range (important for wrap-around colors like red)
    for lower, upper in hsv_ranges:
        mask |= cv.inRange(hsv_frame, lower, upper)

    return mask


def main():
    # -----------------------------
    # Track MULTIPLE colors (each gets its own bbox + label)
    # -----------------------------
    # Replace your single TARGET_BGR / LABEL_TEXT :contentReference[oaicite:2]{index=2}
    TARGETS = [
        {
            "label": "Yellow",
            "bgr": (0, 255, 255),
            "box_color": (0, 255, 0),      # green bbox
            "text_color": (0, 255, 255),   # yellow text
        },
        {
            "label": "Blue",
            "bgr": (255, 0, 0),
            "box_color": (255, 0, 0),      # blue bbox
            "text_color": (255, 0, 0),     # blue text
        },
    ]

    # How strict should the HSV detection be?
    h_tol = 20
    s_min = 100
    v_min = 100

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open camera (VideoCapture(0)).")

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 20.0

    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out_frame = cv.VideoWriter("output.avi", fourcc, float(fps), (w, h))
    out_mask  = cv.VideoWriter("mask.avi",   fourcc, float(fps), (w, h))

    # Precompute HSV ranges for each target color (do once)
    for t in TARGETS:
        t["hsv_ranges"] = get_hsv_ranges_for_bgr(
            t["bgr"], h_tol=h_tol, s_min=s_min, v_min=v_min
        )

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read a frame from camera. Stopping.")
                break

            frame = cv.flip(frame, 1)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            # We'll keep ONE combined diagnostic mask video,
            # but still detect/draw per-color bboxes.
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for t in TARGETS:
                # Build a mask for this specific color
                mask_i = build_color_mask(hsv, t["hsv_ranges"])

                # Same cleanup you already do :contentReference[oaicite:3]{index=3}
                mask_i = cv.morphologyEx(mask_i, cv.MORPH_OPEN, kernel)
                mask_i = cv.morphologyEx(mask_i, cv.MORPH_CLOSE, kernel)

                # Add into combined diagnostic mask
                combined_mask |= mask_i

                # Find bbox for THIS color
                coords = cv.findNonZero(mask_i)
                if coords is None:
                    continue

                x, y, bw, bh = cv.boundingRect(coords)

                # Draw bbox
                cv.rectangle(frame, (x, y), (x + bw, y + bh), t["box_color"], 3)

                # Label above bbox (same idea as your Yellow label code)
                text_x = x
                text_y = max(0, y - 10)

                (text_w, text_h), baseline = cv.getTextSize(
                    t["label"], font, font_scale, thickness
                )
                bg_tl = (text_x, max(0, text_y - text_h - baseline))
                bg_br = (text_x + text_w, text_y + baseline)

                cv.rectangle(frame, bg_tl, bg_br, (0, 0, 0), -1)
                cv.putText(
                    frame, t["label"], (text_x, text_y),
                    font, font_scale, t["text_color"], thickness, cv.LINE_AA
                )

                # Optional centroid
                cx = x + bw // 2
                cy = y + bh // 2
                cv.circle(frame, (cx, cy), 5, t["box_color"], -1)

            # Write outputs
            out_frame.write(frame)
            out_mask.write(cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR))

            # Live windows
            cv.imshow("Frame (Tracked)", frame)
            cv.imshow("Mask (Diagnostic)", combined_mask)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        out_frame.release()
        out_mask.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
