# object_detect_v4.py
#
# Edited version of object_detect_v3.py. Two new capabilities:
#
#   1. SHOW_LABELS toggle. When False, boxes are still drawn but the device
#      name + confidence text (and its colored label bar) are not. Flip it to
#      True when you want the labels back for debugging.
#
#   2. Duplicate-box suppression. The model sometimes reports two boxes for the
#      same device, stacked almost on top of each other, which makes the count
#      flicker (e.g. Flathead_Block jumping between 1 and 2). After detection we
#      now keep only the strongest box when two boxes of the SAME class overlap
#      too much, measured with IoU (see oriented_box_iou below). The count table
#      reads from this filtered list too, so the table and the overlay always
#      agree.
#
# OBB (oriented bounding box) notes carried over from v3:
#
#   - An OBB model stores detections in  results[0].obb  (NOT .boxes).
#   - Each oriented box is FOUR corner points (a rotated rectangle), so we read
#     det.xyxyxyxy, flip the corners back into the operator's orientation, and
#     draw with cv2.polylines instead of cv2.rectangle.
#
# The Elgato Camera Hub is set to Mirror + Flip so the OPERATOR sees the panel
# the way they expect. The model was trained on the ORIGINAL orientation, so we
# undo the camera's Mirror+Flip before detection, then map the boxes back into
# the operator's orientation for display. No retraining required.
#
# Keyboard controls (focus must be on the video window):
#   q  -> quit
#   c  -> capture a still image (saved in the captures/ folder)
#   r  -> start/stop video recording (saved in the recordings/ folder)
#
# Still images can be captured while a video is being recorded.

import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# --- Settings you might want to change ---

MODEL_PATH = "weights_v2_4_obb.pt"
CAMERA_INDEX = 1
CONFIDENCE_THRESHOLD = 0.40

# Show the device name + confidence text above each box?
#   True  -> draw the label (e.g. "Flathead_Block 0.94")
#   False -> draw the box only, no text
SHOW_LABELS = False

# How much two boxes of the SAME class may overlap before we treat the weaker
# one as a duplicate and hide it. This is an IoU value:
#   0.0 = the boxes don't touch at all
#   1.0 = the boxes sit perfectly on top of each other
# Raise it toward 0.8 if real, separate devices start disappearing.
# Lower it if duplicates still slip through.
OVERLAP_THRESHOLD = 0.5

CAPTURES_FOLDER = "captures"
RECORDINGS_FOLDER = "recordings"

# How the Camera Hub is transforming the image, expressed as an OpenCV flip:
#     1  = Mirror only        (left-right)
#     0  = Flip only          (top-bottom)
#    -1  = Mirror AND Flip     (180-degree rotation)  <-- current Camera Hub setting
#  None  = no transform (model sees the camera feed directly)
#
# Each of these flips is its own inverse, so the SAME value is used to undo the
# transform (before detection) and to map the boxes back (for the operator).
# If you change the Camera Hub orientation, change this to match.
CAMERA_FLIP_CODE = -1

# Box colors (Blue, Green, Red order), chosen per class so classes look
# different. Extra colors are fine; we just cycle through the list.
BOX_COLORS = [
    (0, 255, 0),     # green
    (255, 128, 0),   # blue-ish
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (0, 128, 255),   # orange
    (255, 255, 0),   # cyan
]

# --- Helper functions ---

def make_timestamped_filename(folder, prefix, extension):
    """Return a path like 'captures/image_2026-05-12_14-30-22.jpg'."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{timestamp}.{extension}"
    return os.path.join(folder, filename)


def flip_point(x, y, flip_code, width, height):
    """
    Move a single (x, y) point between the model's (native) orientation and the
    operator's orientation. The model produces boxes in native coordinates; we
    use this to place those boxes correctly on the operator-oriented frame.
    """
    if flip_code == 1:        # mirror (left-right)
        return width - 1 - x, y
    elif flip_code == 0:      # flip (top-bottom)
        return x, height - 1 - y
    elif flip_code == -1:     # mirror + flip (180 degrees)
        return width - 1 - x, height - 1 - y
    else:                     # no transform
        return x, y


def oriented_box_iou(corners_a, corners_b):
    """
    Measure how much two oriented (rotated) boxes overlap, as a number from
    0.0 (no overlap) to 1.0 (identical boxes). This is the standard 'IoU'
    score: the area where the boxes overlap divided by the total area they
    cover together.

    'corners_a' and 'corners_b' are each a list of four (x, y) points.
    """
    # OpenCV wants each box described as (center, (width, height), angle).
    # cv2.minAreaRect builds that description from the four corner points.
    box_a = cv2.minAreaRect(np.array(corners_a, dtype=np.float32))
    box_b = cv2.minAreaRect(np.array(corners_b, dtype=np.float32))

    # Find the region where the two boxes intersect. OpenCV returns the
    # corner points of that overlapping region (or None if they don't touch).
    overlap_type, overlap_points = cv2.rotatedRectangleIntersection(box_a, box_b)

    if overlap_points is None or len(overlap_points) == 0:
        return 0.0  # the boxes don't overlap at all

    # Area of the overlapping region. convexHull puts the points in order so
    # the area comes out correct.
    overlap_polygon = cv2.convexHull(overlap_points)
    intersection_area = cv2.contourArea(overlap_polygon)

    # Area of each box is just width * height from its description above.
    area_a = box_a[1][0] * box_a[1][1]
    area_b = box_b[1][0] * box_b[1][1]

    union_area = area_a + area_b - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def extract_detections(yolo_result, class_names, flip_code, width, height):
    """
    Pull every oriented detection out of the YOLO result and turn each one into
    a simple record we can work with later: its four corners (already flipped
    into the operator's orientation), its confidence, and its class.

    Returns a list of dicts. Empty list if nothing was detected.
    """
    records = []

    # OBB models store detections in .obb, not .boxes.
    detections = yolo_result.obb
    if detections is None or len(detections) == 0:
        return records

    for det in detections:
        # det.xyxyxyxy[0] is the four corner points [[x1,y1],...,[x4,y4]]
        # in native (un-flipped) pixel coordinates.
        corners_native = det.xyxyxyxy[0].tolist()

        # Flip every corner from native back into the operator's orientation.
        corners_operator = [
            flip_point(x, y, flip_code, width, height)
            for (x, y) in corners_native
        ]
        corners_operator = [(int(x), int(y)) for (x, y) in corners_operator]

        class_index = int(det.cls[0])
        records.append({
            "corners": corners_operator,
            "confidence": float(det.conf[0]),
            "class_index": class_index,
            "class_name": class_names[class_index],
        })

    return records


def remove_overlapping_duplicates(records, overlap_threshold):
    """
    When two boxes of the SAME class overlap more than 'overlap_threshold',
    keep only the more confident one and drop the other.

    How it works (this is the classic 'greedy' approach):
      1. Sort the detections strongest-confidence-first.
      2. Walk down the list. Keep a box unless it overlaps an already-kept box
         of the same class by more than the threshold.

    Boxes of DIFFERENT classes never suppress each other, so a Terminal_1
    sitting near a Terminal_2 is left alone.
    """
    records_sorted = sorted(records, key=lambda r: r["confidence"], reverse=True)

    kept = []
    for candidate in records_sorted:
        is_duplicate = False
        for keeper in kept:
            # Only compare boxes of the same class.
            if keeper["class_index"] != candidate["class_index"]:
                continue
            if oriented_box_iou(keeper["corners"], candidate["corners"]) > overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)

    return kept


def draw_one_detection(frame, corners, label, color):
    """
    Draw a single ORIENTED (rotated) bounding box, plus an upright label if
    'label' is given. Pass label=None to draw the box only (no text).

    'corners' is a list of four (x, y) points describing the rotated rectangle.
    cv2.polylines connects them in order and closes the loop, so the box can sit
    at any angle.
    """
    points = np.array(corners, dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

    # No label requested -> we're done after drawing the box.
    if label is None:
        return

    # Anchor the label at the highest corner (smallest y) so it sits on top.
    anchor_x, anchor_y = min(corners, key=lambda p: p[1])

    # A small filled rectangle behind the text keeps the label readable over
    # the busy panel/wood background.
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    label_top = max(anchor_y - text_height - 6, 0)
    cv2.rectangle(frame, (anchor_x, label_top),
                  (anchor_x + text_width + 6, label_top + text_height + 6), color, -1)
    cv2.putText(frame, label, (anchor_x + 3, label_top + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


def draw_detections(frame, records, show_labels):
    """
    Draw every detection record onto the operator-oriented frame. Whether the
    name + confidence text appears is controlled by 'show_labels'.
    """
    for record in records:
        color = BOX_COLORS[record["class_index"] % len(BOX_COLORS)]

        if show_labels:
            label = f"{record['class_name']} {record['confidence']:.2f}"
        else:
            label = None  # box only, no text

        draw_one_detection(frame, record["corners"], label, color)


def count_detections_by_class(records):
    """
    Given the (already filtered) list of detection records, return a dict
    mapping class name -> how many times that class appears.
    """
    counts = {}
    for record in records:
        class_name = record["class_name"]
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def draw_detection_table(frame, counts):
    """
    Draw a small semi-transparent table in the top-right corner of the
    frame showing detection counts per class. Modifies the frame in place.
    """
    # Visual settings. Tweak these if you want bigger/smaller text.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 18  # vertical pixels per row of text
    padding = 8       # pixels of empty space inside the box edge

    # Build the list of text rows we want to display.
    if len(counts) == 0:
        rows = ["No detections"]
    else:
        rows = ["Detections this frame:"]
        # Sort alphabetically so rows don't reorder frame-to-frame.
        for class_name in sorted(counts.keys()):
            rows.append(f"  {class_name}: {counts[class_name]}")

    # Figure out how wide the box needs to be by measuring each row of text
    # and taking the longest. cv2.getTextSize returns the pixel size.
    max_text_width = 0
    for row in rows:
        (text_width, _), _ = cv2.getTextSize(row, font, font_scale, font_thickness)
        if text_width > max_text_width:
            max_text_width = text_width

    box_width = max_text_width + 2 * padding
    box_height = line_height * len(rows) + 2 * padding

    # Position the box in the top-right corner with a small margin.
    frame_h, frame_w = frame.shape[:2]
    x1 = frame_w - box_width - 10
    y1 = 10
    x2 = x1 + box_width
    y2 = y1 + box_height

    # Draw a semi-transparent black background so text is readable but
    # detections behind the table aren't fully hidden.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw each row of text on top of the blended background.
    for i, row in enumerate(rows):
        text_x = x1 + padding
        text_y = y1 + padding + line_height * (i + 1) - 4
        cv2.putText(frame, row, (text_x, text_y),
                    font, font_scale, (255, 255, 255), font_thickness)




# ----------------------------------------------------------------------
# Standalone runner
#
# This block only runs when you launch this file directly
# (python object_detection.py). When main.py IMPORTS this module to reuse
# the functions above, this block does NOT run, so importing does not open
# the camera or start a loop.
# ----------------------------------------------------------------------

def run_standalone():
    # --- Create output folders if they don't already exist ---

    os.makedirs(CAPTURES_FOLDER, exist_ok=True)
    os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

    # --- Load the model ---

    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    # --- Open the camera ---

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

    print(f"Camera resolution: {frame_width} x {frame_height} at {camera_fps:.1f} FPS")
    print("Controls: 'q' quit | 'c' capture image | 'r' start/stop recording")

    # --- Recording state ---

    video_writer = None
    current_video_path = None


    # --- Main loop ---

    while True:
        # The camera feed arrives already Mirror+Flipped (the operator orientation).
        success, camera_frame = camera.read()
        if not success:
            print("Failed to grab a frame. Exiting.")
            break

        height, width = camera_frame.shape[:2]

        # Undo the camera's transform so the model sees its trained orientation.
        # (cv2.flip with the same code undoes it, because each flip is self-inverse.
        #  If CAMERA_FLIP_CODE is None, the frame is used as-is.)
        if CAMERA_FLIP_CODE is None:
            model_frame = camera_frame
        else:
            model_frame = cv2.flip(camera_frame, CAMERA_FLIP_CODE)

        # Run YOLO on the model's (native) orientation.
        results = model.predict(model_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Pull out every detection, flip its corners into the operator's view, then
        # drop duplicate boxes that overlap too much. Everything below uses this
        # one filtered list, so the overlay and the count table always agree.
        all_detections = extract_detections(
            results[0], model.names, CAMERA_FLIP_CODE, width, height)
        detections = remove_overlapping_duplicates(all_detections, OVERLAP_THRESHOLD)

        # Build the operator view: start from the ORIGINAL camera frame (already in
        # the operator's orientation) and draw the surviving detections onto it.
        annotated_frame = camera_frame.copy()
        draw_detections(annotated_frame, detections, SHOW_LABELS)

        # Count detections per class (from the filtered list) and draw the table.
        counts = count_detections_by_class(detections)
        draw_detection_table(annotated_frame, counts)

        # Write to the video file if currently recording.
        if video_writer is not None:
            video_writer.write(annotated_frame)

        # Make a separate copy for the on-screen preview. The REC indicator
        # is added here only, so it appears in the window but NOT in saved files.
        display_frame = annotated_frame.copy()

        if video_writer is not None:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Object Detection (Operator View)", display_frame)

        # --- Handle keypresses ---

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            image_path = make_timestamped_filename(CAPTURES_FOLDER, "image", "jpg")
            cv2.imwrite(image_path, annotated_frame)
            print(f"Saved image: {image_path}")

        elif key == ord('r'):
            if video_writer is None:
                current_video_path = make_timestamped_filename(
                    RECORDINGS_FOLDER, "video", "mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    current_video_path,
                    fourcc,
                    camera_fps,
                    (frame_width, frame_height),
                )
                print(f"Started recording: {current_video_path}")
            else:
                video_writer.release()
                video_writer = None
                print(f"Saved recording: {current_video_path}")

    # --- Clean up ---

    if video_writer is not None:
        video_writer.release()
        print(f"Saved recording: {current_video_path}")

    camera.release()
    cv2.destroyAllWindows()
    print("Stopped.")

if __name__ == "__main__":
    run_standalone()