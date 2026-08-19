# camera_pose.py
#
# Works out WHERE THE CAMERA IS and WHICH WAY IT IS POINTING, relative to the
# plywood work surface, using the four ArUco markers already stuck to it.
#
# This file is a LIBRARY: importing it does nothing except define functions.
# The runnable tool is measure_camera_pose.py. Same pattern as
# object_detection.py and probe_tracking.py in the delivered system.
#
# Like those modules, this one takes its settings as ordinary function
# arguments and does NOT import camera_config, so it can be tested on its own.
#
# ======================================================================
# THE IDEA, IN PLAIN TERMS
# ======================================================================
#
# We know two things:
#   1. Exactly where each marker corner sits on the plywood, in millimetres.
#      (You measured this and typed it into MARKER_LAYOUT.)
#   2. Exactly which pixel each of those corners landed on in the image.
#      (The ArUco detector found them.)
#
# There is only one place a camera can be, and one direction it can face, that
# turns list 1 into list 2. Finding it is called "solving the Perspective-n-
# Point problem", and OpenCV does it with cv2.solvePnP.
#
# The answer comes back as two small vectors, rvec and tvec, which together say
# "to get from board coordinates to camera coordinates, rotate by rvec then
# shift by tvec". Everything else in this file is unpacking that answer into
# numbers a person can act on.
#
# ======================================================================
# THE COORDINATE SYSTEM (all distances in millimetres)
# ======================================================================
#
#       Z  (straight up out of the plywood, toward the camera)
#       |
#       |
#       +-------- X  (to the operator's right)
#      /
#     Y  (away from the operator)
#
# ======================================================================
# WHY THIS FILE REPORTS TILT/AZIMUTH BEFORE YAW/PITCH/ROLL
# ======================================================================
#
# Yaw, pitch and roll are the familiar aircraft angles, and they are reported
# here because they were asked for. But they have a serious flaw for this
# particular job: they are DEGENERATE when the camera looks straight down.
#
# Picture an aeroplane pointing vertically at the ground. Rolling it and
# yawing it now do the same thing - you cannot tell them apart, and the maths
# cannot either. This is called gimbal lock, and it happens at exactly 90
# degrees of pitch, which is exactly your P0 overhead baseline. Near that pose
# the yaw and roll numbers become numerically unstable and will jitter wildly
# between frames even though the camera has not moved.
#
# So the PRIMARY answer this file gives is a different set of three angles that
# has no singularity anywhere you will actually put the camera:
#
#   tilt        how far the camera is off vertical, as seen from a chosen point
#               on the board. 0 = directly overhead. This is theta in the
#               Camera Position Evaluation Plan.
#   azimuth     which direction the camera lies in, seen from that same point.
#               This is phi in the plan.
#   image roll  how rotated the board appears in the picture.
#
# Yaw, pitch and roll are still reported, with a warning attached whenever the
# pose is close enough to vertical that they cannot be trusted.

import itertools
import math

import cv2
import numpy as np

# How far sideways the camera has to be from the reference point before its
# azimuth means anything. Below this we report azimuth as undefined rather than
# as a made-up number. See the note inside describe_pose().
AZIMUTH_DEAD_ZONE_MM = 1.0


# ======================================================================
# Building the board model: where every marker corner is, in millimetres
# ======================================================================

def build_marker_object_points(marker_layout):
    """
    Turn the human-friendly MARKER_LAYOUT description into the arrays OpenCV
    needs: for each marker ID, the 3D positions of its four corners.

    marker_layout : {marker_id: {"center_mm": (x, y),
                                 "size_mm": side length of the black square,
                                 "rotation_deg": 0 / 90 / 180 / 270}}

    Returns {marker_id: (4, 3) float32 array}.

    CORNER ORDER MATTERS. cv2.aruco.detectMarkers always returns a marker's
    corners in this order, walking clockwise as the marker is printed:
        index 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left
    We must list the physical positions in the SAME order, or solvePnP will
    fit a twisted, wrong answer while still reporting a small-ish error.
    """
    object_points = {}

    for marker_id, spec in marker_layout.items():
        center_x, center_y = spec["center_mm"]
        half = spec["size_mm"] / 2.0
        angle = math.radians(spec.get("rotation_deg", 0))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Offsets from the marker centre to each corner, before rotation, in
        # the same order the detector uses. Y is "away from the operator", so
        # the marker's own "top" is toward larger Y.
        offsets = [
            (-half, +half),   # 0 top-left
            (+half, +half),   # 1 top-right
            (+half, -half),   # 2 bottom-right
            (-half, -half),   # 3 bottom-left
        ]

        corners = []
        for offset_x, offset_y in offsets:
            # Standard 2D rotation, anticlockwise seen from above the board.
            rotated_x = offset_x * cos_a - offset_y * sin_a
            rotated_y = offset_x * sin_a + offset_y * cos_a
            corners.append([center_x + rotated_x, center_y + rotated_y, 0.0])

        object_points[int(marker_id)] = np.array(corners, dtype=np.float32)

    return object_points


def marker_centroid(marker_layout):
    """The average of the four marker centres. Used as the default point that
    tilt and azimuth are reported about, because it is close to the middle of
    the working area."""
    centers = [spec["center_mm"] for spec in marker_layout.values()]
    mean_x = sum(c[0] for c in centers) / len(centers)
    mean_y = sum(c[1] for c in centers) / len(centers)
    return (mean_x, mean_y)


# ======================================================================
# Matching what was detected to what we know
# ======================================================================

def collect_correspondences(detected_corners, detected_ids, object_points_by_id):
    """
    Pair up the detected markers with their known physical positions.

    detected_corners   : the list cv2.aruco.detectMarkers returned
    detected_ids       : the ids array it returned (or None)
    object_points_by_id: from build_marker_object_points()

    Returns (object_points, image_points, used_ids). Markers that are not in
    the layout - a stray marker on the bench, say - are ignored rather than
    treated as an error.
    """
    if detected_ids is None or len(detected_ids) == 0:
        return None, None, []

    object_list = []
    image_list = []
    used_ids = []

    for corners, marker_id in zip(detected_corners, detected_ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in object_points_by_id:
            continue
        object_list.append(object_points_by_id[marker_id])
        image_list.append(corners.reshape(4, 2))
        used_ids.append(marker_id)

    if not used_ids:
        return None, None, []

    object_points = np.concatenate(object_list).astype(np.float32)
    image_points = np.concatenate(image_list).astype(np.float32)
    return object_points, image_points, sorted(used_ids)


# ======================================================================
# Solving for the pose
# ======================================================================

def reprojection_error(object_points, image_points, rvec, tvec,
                       camera_matrix, dist_coeffs):
    """
    Take the answer, use it to PREDICT where every corner should have landed,
    and measure how far off those predictions are, in pixels.

    This is the single most useful number for telling whether to believe a
    pose. Under about 1 pixel is good. Over about 3 pixels means something is
    wrong - most often a mistyped marker position or a wrong rotation_deg.
    """
    projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                     camera_matrix, dist_coeffs)
    difference = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))


def solve_pose(object_points, image_points, camera_matrix, dist_coeffs):
    """
    Find the camera pose. Returns (rvec, tvec, error_pixels) or None.

    Two steps, and both matter:

      1. cv2.SOLVEPNP_IPPE gets a first answer. IPPE is a method designed
         specifically for points that all lie on a FLAT plane, which ours do
         (every marker is stuck to the same sheet of plywood). The general
         solver is less reliable in the planar case.

      2. cv2.solvePnPRefineLM then polishes that answer, nudging it until the
         reprojection error is as small as it will go. This typically buys a
         useful fraction of a degree.
    """
    if object_points is None or len(object_points) < 4:
        return None

    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE)
    if not success:
        return None

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec)

    error = reprojection_error(object_points, image_points, rvec, tvec,
                               camera_matrix, dist_coeffs)
    return rvec, tvec, error


# ======================================================================
# Unpacking the pose into numbers a person can use
# ======================================================================

def rotation_matrix_to_zyx_euler(matrix):
    """
    Pull yaw, pitch and roll (in degrees) out of a rotation matrix, using the
    aviation convention: rotate about Z first (yaw), then about the new Y
    (pitch), then about the new X (roll).

    Also returns a flag saying whether we are close to gimbal lock, which is
    where pitch approaches +/- 90 degrees and yaw and roll stop being
    separable. See the note at the top of this file: that happens at exactly
    the overhead camera position.
    """
    # The standard closed-form extraction for R = Rz(yaw) . Ry(pitch) . Rx(roll)
    sin_pitch = -matrix[2, 0]
    sin_pitch = max(-1.0, min(1.0, sin_pitch))      # guard against tiny overshoot
    cos_pitch = math.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)

    pitch = math.degrees(math.atan2(sin_pitch, cos_pitch))

    near_lock = cos_pitch < 1e-3
    if near_lock:
        # Fully locked. Yaw and roll are indistinguishable; fold everything
        # into yaw and set roll to zero so the numbers at least stay finite.
        yaw = math.degrees(math.atan2(-matrix[0, 1], matrix[1, 1]))
        roll = 0.0
    else:
        yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        roll = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))

    return yaw, pitch, roll, near_lock


def describe_pose(rvec, tvec, reference_point_mm):
    """
    Turn the raw rvec/tvec into a dictionary of readable measurements.

    reference_point_mm : the (x, y) spot on the board that tilt and azimuth are
                         measured about. Tilt only means something relative to
                         a specific place: the camera sits at one angle above
                         the middle of the board and a different angle above
                         its corner.

    Returns a dict. The keys are documented inline below.
    """
    rotation, _ = cv2.Rodrigues(rvec)
    translation = np.array(tvec, dtype=np.float64).reshape(3)

    # ------------------------------------------------------------------
    # 1. WHERE IS THE CAMERA?
    #
    # solvePnP gives us the board's pose in the CAMERA's frame:
    #     point_in_camera = rotation . point_in_board + translation
    # We want the opposite: the camera's position in the BOARD's frame. Set
    # point_in_camera to the origin (the camera is at its own origin) and
    # rearrange:
    #     0 = rotation . camera_in_board + translation
    #     camera_in_board = -rotation_transposed . translation
    # ------------------------------------------------------------------
    camera_in_board = (-rotation.T @ translation).reshape(3)

    reference = np.array([reference_point_mm[0], reference_point_mm[1], 0.0])
    to_camera = camera_in_board - reference
    distance = float(np.linalg.norm(to_camera))

    # ------------------------------------------------------------------
    # 2. TILT AND AZIMUTH OF THE CAMERA'S POSITION
    #
    # Tilt: the angle between "straight up out of the board" and "the direction
    # you would look from the reference point to reach the camera".
    #     0 deg   = camera directly overhead
    #     30 deg  = camera 30 degrees off vertical
    #     90 deg  = camera down at board level (never happens in practice)
    #
    # THIS is the angle that governs probe parallax, because parallax depends
    # on where the camera IS relative to the point being probed, not on which
    # way the camera happens to be aimed.
    #
    # Azimuth: which compass direction the camera lies in, seen from above.
    #     0 deg   = camera is off toward +X (the operator's right)
    #     90 deg  = camera is off toward +Y (the far side of the board)
    #     180 deg = camera is off toward -X (the operator's left)
    #     -90 deg = camera is off toward -Y (the operator's side)
    # ------------------------------------------------------------------
    if distance > 1e-9:
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, to_camera[2] / distance))))
    else:
        tilt = 0.0

    # AZIMUTH IS UNDEFINED WHEN THE CAMERA IS DIRECTLY OVERHEAD.
    #
    # Azimuth answers "which way from here is the camera", and if the camera is
    # straight up there is no such direction - the same way there is no compass
    # bearing to the North Pole once you are standing on it.
    #
    # Numerically, atan2(0, 0) does not fail; it happily returns a number built
    # from whatever floating-point dust is left in the last decimal places. So
    # we check first and return None, rather than handing back a confident
    # meaningless angle that someone might write on a data sheet.
    #
    # AZIMUTH_DEAD_ZONE_MM is how far sideways the camera must be before we
    # call the direction real. One millimetre is far below anything you could
    # position by hand, so this only ever fires at a genuinely vertical mount.
    horizontal_offset = math.hypot(to_camera[0], to_camera[1])
    if horizontal_offset < AZIMUTH_DEAD_ZONE_MM:
        azimuth = None
    else:
        azimuth = math.degrees(math.atan2(to_camera[1], to_camera[0]))

    # ------------------------------------------------------------------
    # 3. WHERE IS THE CAMERA AIMED?
    #
    # A camera looks along its own +Z axis. To express that direction in board
    # coordinates we multiply by the transposed rotation, which works out to
    # simply taking the third ROW of the rotation matrix.
    #
    # This is a DIFFERENT question from question 2. If the camera sits off to
    # one side but is aimed at the middle of the board, its position tilt and
    # its aim tilt are similar. If it is aimed past the board, they diverge.
    # Comparing the two tells you whether the camera is actually pointed at
    # what you think it is pointed at.
    # ------------------------------------------------------------------
    optical_axis_in_board = rotation[2, :]
    # The camera looks DOWN at the board, so this vector has a negative Z.
    # Angle between the aim direction and "straight down into the board":
    aim_tilt = math.degrees(
        math.acos(max(-1.0, min(1.0, -optical_axis_in_board[2]))))
    aim_heading = math.degrees(
        math.atan2(optical_axis_in_board[1], optical_axis_in_board[0]))

    # Where does the optical axis actually strike the board plane? If this is
    # far from the middle of your working area, the camera is mis-aimed.
    if abs(optical_axis_in_board[2]) > 1e-9:
        steps = -camera_in_board[2] / optical_axis_in_board[2]
        aim_point = camera_in_board + steps * optical_axis_in_board
        aim_point_mm = (float(aim_point[0]), float(aim_point[1]))
    else:
        aim_point_mm = None

    # ------------------------------------------------------------------
    # 4. HOW ROTATED DOES THE BOARD LOOK IN THE PICTURE?
    #
    # Take the board's +X axis and see which way it runs across the image.
    # The board's X axis in camera coordinates is the first COLUMN of the
    # rotation matrix; its first two components are its direction on the
    # sensor.
    #
    # Image Y points DOWN, so a positive angle here means the board's X axis
    # runs clockwise from horizontal as you look at the screen.
    #
    # This matters practically: the YOLO model was trained with the board at
    # one particular rotation. If this number drifts far from its calibration
    # value, expect detection to get worse - and that is a dataset problem, not
    # a camera-position problem.
    # ------------------------------------------------------------------
    board_x_in_camera = rotation[:, 0]
    image_roll = math.degrees(math.atan2(board_x_in_camera[1],
                                         board_x_in_camera[0]))

    # ------------------------------------------------------------------
    # 5. YAW, PITCH, ROLL (the aviation angles)
    #
    # To get numbers that read the way a pilot would expect, we first rewrite
    # the camera's axes in an aviation-style frame:
    #     body X = forward  (the camera's optical axis, its +Z)
    #     body Y = right    (the camera's +X)
    #     body Z = down     (the camera's +Y, since image Y points down)
    # and an aviation-style world frame with Z pointing DOWN into the board:
    #     world X = board +X, world Y = board -Y, world Z = board -Z
    # (that swap is a 180-degree turn about X, so it stays right-handed and
    #  does not secretly mirror anything).
    # ------------------------------------------------------------------
    forward = rotation[2, :]        # camera +Z, in board coordinates
    right = rotation[0, :]          # camera +X, in board coordinates
    down = rotation[1, :]           # camera +Y, in board coordinates

    def board_to_aviation_world(vector):
        return np.array([vector[0], -vector[1], -vector[2]])

    body_matrix = np.column_stack([
        board_to_aviation_world(forward),
        board_to_aviation_world(right),
        board_to_aviation_world(down),
    ])
    yaw, pitch, roll, near_gimbal_lock = rotation_matrix_to_zyx_euler(body_matrix)

    # Flag the danger zone generously. Within 5 degrees of vertical the yaw and
    # roll numbers are already noticeably twitchy even though the formula has
    # not formally broken down.
    gimbal_warning = near_gimbal_lock or abs(abs(pitch) - 90.0) < 5.0

    return {
        # --- position ---
        "camera_x_mm": float(camera_in_board[0]),
        "camera_y_mm": float(camera_in_board[1]),
        "camera_z_mm": float(camera_in_board[2]),
        "reference_point_mm": (float(reference_point_mm[0]),
                               float(reference_point_mm[1])),
        "distance_mm": distance,

        # --- the recommended angle set (no singularity overhead) ---
        "tilt_deg": tilt,
        "azimuth_deg": azimuth,
        "image_roll_deg": image_roll,

        # --- where the camera is aimed ---
        "aim_tilt_deg": aim_tilt,
        "aim_heading_deg": aim_heading,
        "aim_point_mm": aim_point_mm,

        # --- aviation angles, for completeness ---
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "gimbal_lock_warning": bool(gimbal_warning),
    }


# ======================================================================
# Predicting probe parallax from the measured pose
# ======================================================================

def parallax_offset(rvec, tvec, board_point_mm, height_mm):
    """
    Predict how far the probe's tracked colour will APPEAR to be from where
    the probe tip actually is, at a given spot on the board.

    board_point_mm : (x, y) - where the probe tip really is
    height_mm      : how high above the board the tracked colour sits

    Returns (offset_x_mm, offset_y_mm, magnitude_mm).

    HOW THIS WORKS
      The tracked colour is at the point directly above the tip, height_mm up.
      The camera sees it along a straight line from the camera to that raised
      point. The system, which assumes everything lies flat on the board, will
      report wherever that line CROSSES the board surface. The gap between that
      crossing point and the true tip is the error.

      This is exact, not the tan(theta) approximation from the test plan. The
      approximation assumes the camera is infinitely far away; this uses where
      the camera actually is, so it correctly shows the error growing toward
      the edges of the board even when the camera is mounted dead overhead.
    """
    rotation, _ = cv2.Rodrigues(rvec)
    translation = np.array(tvec, dtype=np.float64).reshape(3)
    camera = (-rotation.T @ translation).reshape(3)

    tip = np.array([board_point_mm[0], board_point_mm[1], 0.0])
    raised = tip + np.array([0.0, 0.0, height_mm])

    # The camera must be above the colour, or the ray never comes back down.
    if camera[2] - height_mm <= 1e-6:
        return None

    # Walk from the camera toward the raised point and keep going until Z = 0.
    steps = camera[2] / (camera[2] - height_mm)
    crossing = camera + steps * (raised - camera)

    offset_x = float(crossing[0] - tip[0])
    offset_y = float(crossing[1] - tip[1])
    return offset_x, offset_y, float(math.hypot(offset_x, offset_y))


# ======================================================================
# Working out the marker rotations automatically
# ======================================================================

def solve_marker_rotations(detected_corners, detected_ids, marker_layout,
                           camera_matrix, dist_coeffs):
    """
    If you are not sure which way round each marker was stuck to the plywood,
    this tries every combination of 0 / 90 / 180 / 270 degrees for every marker
    and returns the one that fits the image best.

    With four markers that is 4 x 4 x 4 x 4 = 256 combinations. Each one is a
    fast solve, so the whole search takes well under a second.

    Returns (best_rotations_dict, best_error_pixels) or None.

    The winning combination should fit far better than the runners-up - a
    typical good fit is under 1 pixel while a wrong one is 10 or more. If the
    best and second-best are close together, the search has not really decided
    anything and you should measure the markers properly instead.
    """
    marker_ids = sorted(marker_layout.keys())
    best = None
    results = []

    for combination in itertools.product([0, 90, 180, 270],
                                         repeat=len(marker_ids)):
        trial_layout = {}
        for marker_id, rotation_deg in zip(marker_ids, combination):
            spec = dict(marker_layout[marker_id])
            spec["rotation_deg"] = rotation_deg
            trial_layout[marker_id] = spec

        object_points_by_id = build_marker_object_points(trial_layout)
        object_points, image_points, used = collect_correspondences(
            detected_corners, detected_ids, object_points_by_id)
        if not used:
            continue

        solution = solve_pose(object_points, image_points,
                              camera_matrix, dist_coeffs)
        if solution is None:
            continue

        _, _, error = solution
        rotations = dict(zip(marker_ids, combination))
        results.append((error, rotations))
        if best is None or error < best[0]:
            best = (error, rotations)

    if best is None:
        return None

    results.sort(key=lambda item: item[0])
    best_error, best_rotations = results[0]
    runner_up_error = results[1][0] if len(results) > 1 else float("inf")

    return {
        "rotations": best_rotations,
        "error_pixels": best_error,
        "runner_up_error_pixels": runner_up_error,
        "decisive": runner_up_error > best_error * 3.0,
    }


# ======================================================================
# Drawing
# ======================================================================

def draw_board_axes(image, rvec, tvec, camera_matrix, dist_coeffs, length_mm):
    """Draw the board's X (red), Y (green) and Z (blue) axes at the origin, so
    you can see at a glance that the coordinate system is the one you think it
    is. If the blue Z arrow does not point up out of the plywood toward the
    camera, your marker layout has a sign error."""
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length_mm, 3)


def format_pose_report(pose, used_ids, error_pixels):
    """Build the multi-line text report. Returned as a list of lines so the
    caller can print it, draw it on screen, or write it to a file."""
    lines = [
        "CAMERA POSE (board coordinates, millimetres)",
        f"  Markers used          : {used_ids}",
        f"  Reprojection error    : {error_pixels:.3f} px",
        "",
        f"  Camera position       : X {pose['camera_x_mm']:8.1f}   "
        f"Y {pose['camera_y_mm']:8.1f}   Z {pose['camera_z_mm']:8.1f}",
        f"  Distance to reference : {pose['distance_mm']:.1f} mm "
        f"(reference point {pose['reference_point_mm'][0]:.0f}, "
        f"{pose['reference_point_mm'][1]:.0f})",
        "",
        "  RECOMMENDED ANGLES (stable at every pose)",
        f"    Tilt off vertical   : {pose['tilt_deg']:7.2f} deg    "
        f"(theta in the test plan)",
        (f"    Azimuth             : {pose['azimuth_deg']:7.2f} deg    "
         f"(phi in the test plan)")
        if pose["azimuth_deg"] is not None else
        "    Azimuth             : undefined - the camera is directly "
        "overhead, so there is no direction to report",
        f"    Board roll in image : {pose['image_roll_deg']:7.2f} deg",
        "",
        "  WHERE THE CAMERA IS AIMED",
        f"    Aim tilt            : {pose['aim_tilt_deg']:7.2f} deg",
        f"    Aim heading         : {pose['aim_heading_deg']:7.2f} deg",
    ]
    if pose["aim_point_mm"] is not None:
        lines.append(f"    Axis meets board at : "
                     f"({pose['aim_point_mm'][0]:.0f}, "
                     f"{pose['aim_point_mm'][1]:.0f}) mm")
    lines += [
        "",
        "  AVIATION ANGLES",
        f"    Yaw                 : {pose['yaw_deg']:7.2f} deg",
        f"    Pitch               : {pose['pitch_deg']:7.2f} deg",
        f"    Roll                : {pose['roll_deg']:7.2f} deg",
    ]
    if pose["gimbal_lock_warning"]:
        lines += [
            "    !! GIMBAL LOCK WARNING: the camera is within a few degrees of",
            "       straight down, so yaw and roll above are NOT separable and",
            "       will jitter between frames. Quote tilt, azimuth and board",
            "       roll instead - those stay well behaved here.",
        ]
    return lines
