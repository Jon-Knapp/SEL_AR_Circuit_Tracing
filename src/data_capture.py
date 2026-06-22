# capture_data.py
#
# Minimal camera recording program for dataset capture.
# No YOLO, no detection, no overlays — records the raw camera feed.
#
# Use this to film new examples with occlusions, different lighting, and
# varied board positions so you can add them to your training dataset.
#
# Keyboard controls (focus must be on the video window):
#   r  -> start/stop recording  (saved in captures/recordings/)
#   c  -> save a single still   (saved in captures/stills/)
#   q  -> quit

import os
import cv2
from datetime import datetime

# ── Settings ──────────────────────────────────────────────────────────────────

CAMERA_INDEX = 1

RECORDINGS_FOLDER = "captures/testing"
STILLS_FOLDER = "captures/stills"

# IMPORTANT: keep this value the same as in object_detection.py.
# Captured frames are flipped back to native orientation before being saved,
# so they match the orientation your model was trained on. If these images are
# added to the dataset and retrained, the new training data will be consistent
# with the existing data.
#
#     1  = Mirror only
#     0  = Flip only
#    -1  = Mirror AND Flip  <-- current Camera Hub setting
#  None  = no transform
CAMERA_FLIP_CODE = -1

# ── Setup ─────────────────────────────────────────────────────────────────────

os.makedirs(RECORDINGS_FOLDER, exist_ok=True)
os.makedirs(STILLS_FOLDER, exist_ok=True)

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print(f"Could not open camera at index {CAMERA_INDEX}.")
    print("Try a different CAMERA_INDEX (0, 1, 2, ...).")
    exit()

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_fps = camera.get(cv2.CAP_PROP_FPS)

if camera_fps <= 0 or camera_fps > 120:
    camera_fps = 30.0

print(f"Camera: {frame_width} x {frame_height} at {camera_fps:.1f} FPS")
print("Controls: 'r' start/stop recording | 'c' save still | 'q' quit")

# ── State ─────────────────────────────────────────────────────────────────────

video_writer = None
current_video_path = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_timestamped_filename(folder, prefix, extension):
    """Return a path like 'captures/stills/still_2026-06-03_14-30-22.jpg'."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(folder, f"{prefix}_{timestamp}.{extension}")


# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    success, camera_frame = camera.read()
    if not success:
        print("Failed to grab a frame. Exiting.")
        break

    # Undo the Camera Hub's Mirror+Flip so the saved image is in the same
    # native orientation as the existing training data.
    if CAMERA_FLIP_CODE is None:
        frame = camera_frame
    else:
        frame = cv2.flip(camera_frame, CAMERA_FLIP_CODE)

    # Write to video file if currently recording.
    if video_writer is not None:
        video_writer.write(frame)

    # Display the clean frame — no overlays.
    cv2.imshow("Data Capture  (r=record  c=still  q=quit)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r'):
        if video_writer is None:
            current_video_path = make_timestamped_filename(
                RECORDINGS_FOLDER, "recording", "mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                current_video_path, fourcc, camera_fps, (frame_width, frame_height)
            )
            print(f"Recording started:  {current_video_path}")
        else:
            video_writer.release()
            video_writer = None
            print(f"Recording saved:    {current_video_path}")

    elif key == ord('c'):
        still_path = make_timestamped_filename(STILLS_FOLDER, "still", "jpg")
        cv2.imwrite(still_path, frame)
        print(f"Still saved:        {still_path}")

# ── Cleanup ───────────────────────────────────────────────────────────────────

if video_writer is not None:
    video_writer.release()
    print(f"Recording saved:    {current_video_path}")

camera.release()
cv2.destroyAllWindows()
print("Stopped.")