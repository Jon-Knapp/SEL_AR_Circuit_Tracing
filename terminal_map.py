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
# Keeping instance numbers stable across re-detections
# ======================================================================
#
# THE PROBLEM THIS SOLVES
#   The old code numbered devices by their position in a sorted list: the
#   leftmost block was instance 1, the next one instance 2, and so on. That
#   means a number describes a RANK, not a physical object. If two blocks are
#   nearly level with each other, a few pixels of detector jitter flips their
#   order and their numbers swap. Worse, if the detector misses one block on a
#   re-run, every block after it slides down a number - and every connection
#   already recorded against those names now points at the wrong screw.
#
# THE FIX
#   Remember WHERE each numbered device was. On a re-detect, match each new
#   box to the nearest device we already numbered. A block that has not moved
#   (or has only moved a little) keeps its number for the whole session. A
#   block we have never seen before gets the next unused number. A block that
#   disappears simply does not appear this round - its number is NOT handed to
#   somebody else.
#
#   Sorting is now used for exactly one thing: putting brand-new devices in a
#   sensible reading order the first time they are numbered.

class InstanceRegistry:
    """Remembers which physical device owns which instance number."""

    def __init__(self, max_match_distance, row_tolerance):
        """
        max_match_distance : how far (pixels) a device may be from where its
                             number was last seen and still be recognised as
                             the same device.
        row_tolerance      : devices within this many pixels vertically count
                             as the same row when numbering new devices.
        """
        self.max_match_distance = max_match_distance
        self.row_tolerance = row_tolerance

        # class_name -> {instance_number: (center_x, center_y)}
        self.known = {}

    def _row_major_key(self, center):
        """Sort key that gives a real reading order: group into rows first,
        then left-to-right inside each row. Dividing y by the row tolerance and
        rounding turns 'roughly the same height' into 'exactly the same row
        number', which is what makes the y comparison actually work."""
        x, y = center
        return (round(y / self.row_tolerance), x)

    def assign(self, class_name, detections):
        """
        Give every detection of one class its instance number.

        Returns a list of (instance_number, detection) pairs, in the same order
        the detections came in.
        """
        known = self.known.setdefault(class_name, {})
        centers = [box_center(d["corners"]) for d in detections]

        # --- Step 1: match new boxes to devices we have already numbered ---
        # Build every plausible (new box, existing number) pairing along with
        # how far apart they are, then take them cheapest-first. Greedy nearest
        # matching: simple to read, and with a handful of devices it gives the
        # same answer as anything cleverer.
        candidates = []
        for detection_index, (cx, cy) in enumerate(centers):
            for number, (kx, ky) in known.items():
                distance = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
                if distance <= self.max_match_distance:
                    candidates.append((distance, detection_index, number))
        candidates.sort()

        number_for_detection = {}
        numbers_used = set()
        for distance, detection_index, number in candidates:
            if detection_index in number_for_detection:
                continue                 # this box already claimed a number
            if number in numbers_used:
                continue                 # this number already went to a box
            number_for_detection[detection_index] = number
            numbers_used.add(number)

        # --- Step 2: hand out fresh numbers to anything left over ---
        leftovers = [i for i in range(len(detections))
                     if i not in number_for_detection]
        leftovers.sort(key=lambda i: self._row_major_key(centers[i]))

        next_number = max(known.keys(), default=0)
        for detection_index in leftovers:
            next_number += 1
            number_for_detection[detection_index] = next_number
            print(f"[instances] {class_name}: new device -> instance "
                  f"{next_number}")

        # --- Step 3: remember where every numbered device is NOW ---
        # This lets a device drift slowly across the session (or be nudged) and
        # still be recognised next time.
        for detection_index, number in number_for_detection.items():
            known[number] = centers[detection_index]

        # --- Step 4: say so out loud if a numbered device went missing ---
        # Its number is deliberately NOT reused, but any connection already
        # recorded against it can no longer be shown on the board, so the
        # operator needs to know.
        missing = sorted(set(known) - set(number_for_detection.values()))
        if missing:
            print(f"[instances] WARNING {class_name}: instance(s) {missing} "
                  f"were not found in this detection. Their numbers are held "
                  f"in reserve, but any connection recorded on them is no "
                  f"longer displayed.")

        return [(number_for_detection[i], detections[i])
                for i in range(len(detections))]


# ======================================================================
# Applying a calibrated template to live detections (the run-time step)
# ======================================================================

def apply_template(detections, template_devices, registry):
    """
    For every detected device, look up its class's stored template and rebuild
    that device's terminal points in current pixel coordinates.

      detections       : detection records (each has 'class_name', 'class_index',
                         'corners').
      template_devices : the 'devices' section of a loaded terminal map.
      registry         : an InstanceRegistry that keeps instance numbers stable
                         across re-detections. Create ONE at start-up and reuse
                         it for the whole session - do NOT create a new one
                         each time you re-detect, or instance numbers will
                         reset and this fix does nothing.

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
        # Number the devices FIRST, so numbering does not depend on whether
        # this class happens to have a calibrated template.
        numbered = registry.assign(class_name, class_detections)

        device = template_devices.get(class_name)
        if not device:
            continue                          # this class was never calibrated

        for instance, detection in numbered:
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
