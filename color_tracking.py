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
    # Choose the tape color to track (BGR!)
    # -----------------------------
    # OpenCV uses BGR order, not RGB.
    # (0, 255, 255) is bright yellow in BGR.
    TARGET_BGR = (0, 255, 255)
    LABEL_TEXT = "Yellow"  # <--- what we will draw above the bbox

    # How strict should the HSV detection be?
    # - h_tol: hue tolerance (bigger = more forgiving, but more false positives)
    # - s_min/v_min: minimum saturation/value to avoid picking up gray/dark junk
    h_tol = 20
    s_min = 100
    v_min = 100

    # Morphology settings (helps reduce speckles and fill tiny holes)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    # -----------------------------
    # Open camera
    # -----------------------------
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open camera (VideoCapture(0)).")

    # Get real camera frame size (so writers match frames)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Get FPS if camera reports it; otherwise fall back to a sane default
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 20.0

    # -----------------------------
    # Setup video writers
    # -----------------------------
    # Note: Different machines like different codecs. XVID is common for AVI.
    fourcc = cv.VideoWriter_fourcc(*"XVID")

    out_frame = cv.VideoWriter("output.avi", fourcc, float(fps), (w, h))

    # Most compatible approach:
    # - keep writer as color (3-channel),
    # - convert mask (1-channel) -> BGR before writing.
    out_mask = cv.VideoWriter("mask.avi", fourcc, float(fps), (w, h))

    # Precompute HSV ranges for our target color (cheap, do once)
    hsv_ranges = get_hsv_ranges_for_bgr(TARGET_BGR, h_tol=h_tol, s_min=s_min, v_min=v_min)

    try:
        while True:
            # -----------------------------
            # Read one frame
            # -----------------------------
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read a frame from camera. Stopping.")
                break

            # Optional: mirror view (often feels more natural when holding objects up)
            frame = cv.flip(frame, 1)

            # -----------------------------
            # Convert to HSV and threshold by color
            # -----------------------------
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = build_color_mask(hsv, hsv_ranges)

            # Clean up mask:
            # - OPEN removes small white specks (noise)
            # - CLOSE fills small black holes inside the detected region
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

            # -----------------------------
            # Find a bounding box around the detected region
            # -----------------------------
            # findNonZero returns None if there are zero white pixels in the mask.
            coords = cv.findNonZero(mask)

            if coords is not None:
                x, y, bw, bh = cv.boundingRect(coords)

                # Draw a green rectangle on the original frame
                cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 3)

                # -------- Label "Yellow" above the bbox --------
                # Pick a text position slightly above the top-left corner of the box.
                # Clamp it so it doesn't go off-screen if the box is near the top.
                text_x = x
                text_y = max(0, y - 10)

                # (Optional but recommended) draw a filled rectangle behind the text
                # so the text stays readable on busy backgrounds.
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # Get text size (width, height) so we can size the background rectangle
                (text_w, text_h), baseline = cv.getTextSize(LABEL_TEXT, font, font_scale, thickness)

                # Background rectangle corners:
                # top-left  = (text_x, text_y - text_h - baseline)
                # bottom-right = (text_x + text_w, text_y + baseline)
                # Clamp top-left y to 0 to avoid negative coords
                bg_tl = (text_x, max(0, text_y - text_h - baseline))
                bg_br = (text_x + text_w, text_y + baseline)

                # Draw filled background (black) then text (yellow-ish) on top
                cv.rectangle(frame, bg_tl, bg_br, (0, 0, 0), -1)
                cv.putText(frame, LABEL_TEXT, (text_x, text_y), font, font_scale, (0, 255, 255), thickness, cv.LINE_AA)
                # -----------------------------------------------

                # Optional: show centroid too (nice for debugging)
                cx = x + bw // 2
                cy = y + bh // 2
                cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # -----------------------------
            # Write videos to disk
            # -----------------------------
            out_frame.write(frame)

            # Convert 1-channel mask to 3-channel BGR for max codec/player compatibility
            mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            out_mask.write(mask_bgr)

            # -----------------------------
            # Live debug windows
            # -----------------------------
            cv.imshow("Frame (Tracked)", frame)
            cv.imshow("Mask (Diagnostic)", mask)

            # Quit on 'q' or ESC
            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        # -----------------------------
        # Cleanup (always runs)
        # -----------------------------
        cap.release()
        out_frame.release()
        out_mask.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
