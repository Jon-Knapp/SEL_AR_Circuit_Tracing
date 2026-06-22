# terminal_map.py
#
# Where are the screw terminals, and what is each one called?
#
# THE BIG IDEA: store each terminal as a position INSIDE its device's own
# oriented bounding box, not as a fixed pixel in the image.
#
#   We describe a point inside a box with two fractions:
#       u  = how far ALONG THE LENGTH of the device      (0 = one end, 1 = other)
#       v  = how far ACROSS THE WIDTH of the device       (0 = one edge, 1 = other)
#   u or v may go slightly below 0 or above 1 - that is fine, and is exactly
#   what the angled green terminals need (their screws sit just outside the box).
#
# Because the box is ORIENTED (it rotates to hug the device), (u, v) rotate with
# the device too. So we calibrate ONE device of each class, store its terminals
# as (u, v), and then at run time we can re-detect EVERY device of that class -
# anywhere on the board, at any angle - and rebuild its terminal positions by
# applying the same (u, v) to that instance's box.
#
# CAVEAT (read me): a rectangle looks the same rotated 180 degrees, so the box
# tells us the device's axis but not reliably which END is which. If a device is
# placed end-for-end versus how it was calibrated, the numbering flips. For a
# device placed the usual way up this never happens; the free-for-all demo
# station is where it could, and fixing it is future work.
#
# Coordinate frame: everything is in RAW operator-frame pixels - the same pixels
# the probe tracker and the detector use.

import json
import os
from datetime import datetime

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ======================================================================
# Box geometry: turning a box + a point into (u, v), and back again
# ======================================================================

def order_box(corners):
    """
    Put the four corners in a fixed order [A, B, C, D] where:
        A -> B  runs along the LONG axis  (the device's length)
        A -> D  runs along the SHORT axis (the device's width)
    so we can measure 'along the length' and 'across the width' cleanly.
    """
    points = np.array(corners, dtype=np.float32)

    # minAreaRect fits the tightest rotated rectangle; boxPoints returns its
    # four corners walking around the rectangle in order.
    rectangle = cv2.minAreaRect(points)
    box = cv2.boxPoints(rectangle)            # p0, p1, p2, p3 (going around)

    edge_0_to_1 = np.linalg.norm(box[1] - box[0])
    edge_1_to_2 = np.linalg.norm(box[2] - box[1])
    if edge_0_to_1 >= edge_1_to_2:
        # p0->p1 is already the long edge.
        ordered = box
    else:
        # Rotate the order by one so the long edge comes first.
        ordered = np.array([box[1], box[2], box[3], box[0]], dtype=np.float32)
    return ordered


def point_to_uv(ordered_box, point):
    """
    Given an ordered box and a pixel point, return (u, v): how far along the
    length and across the width the point sits, as fractions. This is how a
    probe centroid becomes a box-relative terminal location during calibration.
    """
    A = ordered_box[0]
    long_axis = ordered_box[1] - A            # A -> B (length direction)
    short_axis = ordered_box[3] - A           # A -> D (width direction)

    relative = np.array(point, dtype=np.float32) - A

    # Project 'relative' onto each axis and divide by that axis's length. For a
    # rectangle the two axes are perpendicular, so this is exact.
    u = float(np.dot(relative, long_axis) / np.dot(long_axis, long_axis))
    v = float(np.dot(relative, short_axis) / np.dot(short_axis, short_axis))
    return u, v


def uv_to_point(ordered_box, u, v):
    """
    The reverse of point_to_uv: given an ordered box and (u, v), return the
    pixel point. This is how a stored terminal gets placed onto a freshly
    detected device at run time.
    """
    A = ordered_box[0]
    long_axis = ordered_box[1] - A
    short_axis = ordered_box[3] - A
    point = A + u * long_axis + v * short_axis
    return float(point[0]), float(point[1])


def box_center(corners):
    """The middle of a box, used to put multiple boxes of one class in a stable
    left-to-right, top-to-bottom order so instance numbers don't jump around."""
    center = np.array(corners, dtype=np.float32).mean(axis=0)
    return float(center[0]), float(center[1])


# ======================================================================
# Applying a calibrated template to live detections (the run-time step)
# ======================================================================

def apply_template(detections, template_devices):
    """
    For every detected device, look up its class's stored template and rebuild
    that device's terminal points in current pixel coordinates.

      detections       : detection records (each has 'class_name', 'class_index',
                         'corners').
      template_devices : the 'devices' section of a loaded terminal map.

    Returns a flat list of terminal records:
        {
          "class_name": "Terminal_1", "class_index": 2,
          "instance": 1,            # which physical block of that class
          "index": 3,               # the terminal's number within the device
          "terminal_id": "Terminal_1_1_3",
          "x": 812.0, "y": 455.0,
        }
    """
    # Group detections by class so we can number multiple blocks of one class.
    by_class = {}
    for detection in detections:
        by_class.setdefault(detection["class_name"], []).append(detection)

    terminals = []
    for class_name, class_detections in by_class.items():
        device = template_devices.get(class_name)
        if not device:
            continue                          # this class was never calibrated

        # Stable instance order: left-to-right, then top-to-bottom.
        ordered_detections = sorted(class_detections,
                                    key=lambda d: box_center(d["corners"]))

        for instance, detection in enumerate(ordered_detections, start=1):
            ordered = order_box(detection["corners"])
            for terminal in device["terminals"]:
                x, y = uv_to_point(ordered, terminal["u"], terminal["v"])
                terminals.append({
                    "class_name": class_name,
                    "class_index": detection["class_index"],
                    "instance": instance,
                    "index": terminal["index"],
                    "terminal_id": f"{class_name}_{instance}_{terminal['index']}",
                    "x": x,
                    "y": y,
                })
    return terminals


def nearest_terminal(point, terminals, max_distance):
    """
    Return the terminal closest to 'point', but only if it is within
    max_distance pixels. Returns None otherwise, so we never credit a touch to a
    terminal the probe is not really on.
    """
    if not terminals:
        return None

    px, py = point
    closest = None
    closest_distance = float(max_distance)
    for terminal in terminals:
        dx = terminal["x"] - px
        dy = terminal["y"] - py
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= closest_distance:
            closest_distance = distance
            closest = terminal
    return closest


# ======================================================================
# Saving and loading the template (one JSON file)
# ======================================================================

def save_template(path, devices, image_size=None, note=""):
    """
    Write the calibrated template to JSON.

      devices : dict mapping class name -> {
                    "calibration_box": [[x,y], [x,y], [x,y], [x,y]],
                    "terminals": [{"index", "u", "v", "calib_xy"}, ...],
                }
    """
    data = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "image_size": image_size,
        "coordinate_note": ("u = along device length, v = across device width, "
                            "both relative to the oriented bounding box"),
        "devices": devices,
    }
    with open(path, "w") as file:
        json.dump(data, file, indent=2)
    return path


def load_template(path):
    """Read the template back. Returns the 'devices' dict, or an empty dict if
    the file does not exist yet."""
    if not os.path.exists(path):
        return {}
    with open(path) as file:
        data = json.load(file)
    return data.get("devices", {})


# ======================================================================
# Drawing
# ======================================================================

def color_for_class(class_index, color_list):
    """Pick the same color the detector uses for this class's box, so a
    terminal's circle matches its device's box."""
    return color_list[class_index % len(color_list)]


def draw_terminal_circle(frame, point, color, highlight=False):
    """Draw ONE terminal as a clean circle in the class color. No text - the
    live and record views stay uncluttered. 'highlight' draws it a little bigger
    and thicker, used to show the terminal a probe is currently on."""
    x, y = int(point[0]), int(point[1])
    radius = 15 if highlight else 9
    thickness = 3 if highlight else 2
    cv2.circle(frame, (x, y), radius, color, thickness)


def draw_terminal_label(frame, point, text, color):
    """Draw a terminal WITH its ID text. Used only on the separate legend image
    (the one you keep next to the JSON / database records), never on the live
    or clean-record views."""
    x, y = int(point[0]), int(point[1])
    cv2.circle(frame, (x, y), 6, color, -1)
    cv2.putText(frame, text, (x + 8, y - 6), FONT, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 8, y - 6), FONT, 0.4, color, 1, cv2.LINE_AA)