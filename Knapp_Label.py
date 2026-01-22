import cv2 as cv
import os
from datetime import datetime

# ---- Settings you can tweak ----
CAMERA_INDEX = 0
LIVE_WINDOW = "Live Camera (press 'c' to capture, 'q' to quit)"
REVIEW_WINDOW = "Review (y=save, n=discard)"

OUTPUT_DIR = "captures"

LABEL_TEXT = "DUMBASS"          # choose a respectful label
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2

# OpenCV uses BGR (not RGB)
TEXT_COLOR = (255, 255, 255)     # white
BG_COLOR = (0, 0, 0)             # black
PADDING = 10                     # padding around text inside background box
MARGIN = 20                      # margin from top-left corner
# --------------------------------


def overlay_label(img, text: str):
    """Draw a solid background rectangle with text on the image (top-left)."""
    out = img.copy()

    (text_w, text_h), baseline = cv.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)

    # Baseline origin for text
    x = MARGIN
    y = MARGIN + text_h

    # Background rectangle coordinates
    x1 = x - PADDING
    y1 = y - text_h - PADDING
    x2 = x + text_w + PADDING
    y2 = y + baseline + PADDING

    # Clamp to image bounds
    h, w = out.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)

    cv.rectangle(out, (x1, y1), (x2, y2), BG_COLOR, thickness=-1)
    cv.putText(out, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return out


def save_image(img):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"capture_{ts}.png")
    ok = cv.imwrite(filename, img)
    if ok:
        print(f"Saved: {filename}")
    else:
        print("Error: Failed to save image.")


def review_frame(labeled_frame):
    """
    Show the labeled frame and let the user accept or discard.
    Returns True if saved, False if discarded.
    """
    cv.imshow(REVIEW_WINDOW, labeled_frame)

    while True:
        key = cv.waitKey(0) & 0xFF  # wait indefinitely for a keypress

        if key in (ord('y'), ord('Y')):
            save_image(labeled_frame)
            cv.destroyWindow(REVIEW_WINDOW)
            return True

        if key in (ord('n'), ord('N'), 27):  # 'n' or ESC
            print("Discarded.")
            cv.destroyWindow(REVIEW_WINDOW)
            return False

        # Any other key -> ignore and keep waiting
        print("Press 'y' to save or 'n' to discard (Esc also discards).")


def main():
    cap = cv.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: failed to read frame from camera.")
                continue

            cv.imshow(LIVE_WINDOW, frame)
            key = cv.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                break

            if key in (ord('c'), ord('C')):
                # Freeze the current frame and label it
                labeled = overlay_label(frame, LABEL_TEXT)

                # Optional: hide the live window during review to make it clear it's "frozen"
                cv.destroyWindow(LIVE_WINDOW)

                # Review / accept / discard
                review_frame(labeled)

                # Restore live window after review
                cv.namedWindow(LIVE_WINDOW, cv.WINDOW_AUTOSIZE)

    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
