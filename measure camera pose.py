# measure_camera_pose.py
#
# Measures where the camera is and which way it points, using the four ArUco
# markers on the plywood.
#
#     python measure_camera_pose.py                 live from the camera
#     python measure_camera_pose.py --image shot.png   from a saved photo
#     python measure_camera_pose.py --solve-rotations  work out marker rotations
#
# CONTROLS (live mode)
#     m   take a MEASUREMENT: average many frames and print the full report
#     p   print the parallax prediction table for this pose
#     s   save the last measurement to a JSON file
#     a   run the marker-rotation search on the current frame
#     q   quit
#
# WHY 'm' AVERAGES INSTEAD OF READING ONE FRAME
#     Corner detection wobbles by a fraction of a pixel from frame to frame
#     because of sensor noise. That wobble turns into a fraction of a degree
#     of pose wobble. Averaging a few dozen frames removes most of it, and the
#     spread across those frames is itself worth knowing: it tells you the
#     repeatability of the measurement, which is the number you should quote
#     when you say "the camera was at 30 degrees".

import argparse
import json
import math
import os
from datetime import datetime

import cv2
import numpy as np

import camera_config as cfg
import camera_pose as cp
from calibrate_camera import load_intrinsics

# How many frames 'm' averages over.
MEASUREMENT_FRAMES = 40


# ======================================================================
# Averaging helpers
# ======================================================================

def circular_mean_and_spread(angles_deg):
    """
    Average a set of ANGLES correctly.

    You cannot just add angles and divide. The average of 179 degrees and
    -179 degrees is 180 (they are almost the same direction), but plain
    arithmetic gives 0, which points the opposite way. The fix is to turn each
    angle into a unit vector, average the vectors, and convert back.

    Returns (mean_degrees, spread_degrees), where spread is a standard
    deviation in degrees.
    """
    radians = np.radians(np.array(angles_deg, dtype=np.float64))
    mean_cos = float(np.mean(np.cos(radians)))
    mean_sin = float(np.mean(np.sin(radians)))

    mean = math.degrees(math.atan2(mean_sin, mean_cos))

    resultant = math.hypot(mean_cos, mean_sin)
    if resultant >= 1.0:
        spread = 0.0
    elif resultant <= 1e-12:
        spread = 180.0                    # completely scattered
    else:
        spread = math.degrees(math.sqrt(-2.0 * math.log(resultant)))
    return mean, spread


def average_poses(pose_list):
    """
    Average a list of per-frame pose dictionaries.

    Distances are averaged normally. Angles use the circular average above.
    Alongside each average we keep the spread, so the report can show how
    repeatable the measurement was.
    """
    distance_keys = ["camera_x_mm", "camera_y_mm", "camera_z_mm", "distance_mm"]
    angle_keys = ["tilt_deg", "azimuth_deg", "image_roll_deg",
                  "aim_tilt_deg", "aim_heading_deg",
                  "yaw_deg", "pitch_deg", "roll_deg"]

    averaged = {}
    spreads = {}

    for key in distance_keys:
        values = [pose[key] for pose in pose_list]
        averaged[key] = float(np.mean(values))
        spreads[key] = float(np.std(values))

    for key in angle_keys:
        # Azimuth comes back as None when the camera is directly overhead,
        # where the direction genuinely does not exist. Drop those frames
        # rather than averaging None into a number.
        values = [pose[key] for pose in pose_list if pose[key] is not None]
        if not values:
            averaged[key] = None
            spreads[key] = None
            continue
        mean, spread = circular_mean_and_spread(values)
        averaged[key] = mean
        spreads[key] = spread

    # Non-numeric fields: take them from the last frame, they do not vary.
    averaged["reference_point_mm"] = pose_list[-1]["reference_point_mm"]
    averaged["aim_point_mm"] = pose_list[-1]["aim_point_mm"]
    averaged["gimbal_lock_warning"] = any(p["gimbal_lock_warning"]
                                          for p in pose_list)
    return averaged, spreads


# ======================================================================
# Reporting
# ======================================================================

def print_parallax_table(rvec, tvec, height_mm, sample_points):
    """
    Print the predicted probe parallax at several places on the board.

    This is the number that decides whether a camera position is usable. Read
    it against HALF the spacing between neighbouring terminals: if the error is
    bigger than that, the system will confidently name the wrong terminal.
    """
    print()
    print("PREDICTED PROBE PARALLAX")
    print(f"  Tracked colour height above the board: {height_mm:.1f} mm")
    print()
    print(f"  {'where':<14} {'board (x, y) mm':<20} {'offset mm':<12} "
          f"{'direction':<20}")
    print("  " + "-" * 68)

    for name, point in sample_points:
        result = cp.parallax_offset(rvec, tvec, point, height_mm)
        if result is None:
            print(f"  {name:<14} camera is not above the probe colour")
            continue
        offset_x, offset_y, magnitude = result
        direction = f"({offset_x:+.2f}, {offset_y:+.2f})"
        print(f"  {name:<14} ({point[0]:.0f}, {point[1]:.0f})"
              f"{'':<{max(0, 20 - len(f'({point[0]:.0f}, {point[1]:.0f})'))}}"
              f"{magnitude:>7.2f}     {direction:<20}")

    print()
    print("  Compare the largest value against HALF your tightest terminal")
    print("  pitch. Above that, expect wrong-terminal identifications.")
    print()
    print("  Note that the offsets differ from place to place even with the")
    print("  camera mounted dead overhead. That is not a bug: a terminal near")
    print("  the edge of the frame is being viewed at an angle even when the")
    print("  camera body is perfectly vertical.")
    print()


def print_measurement(averaged, spreads, used_ids, error_pixels, frames):
    lines = cp.format_pose_report(averaged, used_ids, error_pixels)
    print()
    print("=" * 70)
    for line in lines:
        print(line)
    print()
    print(f"  Averaged over {frames} frames. Frame-to-frame spread "
          f"(1 standard deviation):")
    azimuth_spread = ("n/a" if spreads["azimuth_deg"] is None
                      else f"{spreads['azimuth_deg']:.3f} deg")
    print(f"    tilt {spreads['tilt_deg']:.3f} deg   "
          f"azimuth {azimuth_spread}   "
          f"roll {spreads['image_roll_deg']:.3f} deg   "
          f"distance {spreads['distance_mm']:.2f} mm")
    print("    Quote these as the repeatability of the measurement. If the")
    print("    spread is larger than about 0.2 degrees, something is moving:")
    print("    check the camera mount, the lighting, and the markers.")
    print("=" * 70)


def save_measurement(folder, averaged, spreads, used_ids, error_pixels, frames):
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(folder, f"pose_{stamp}.json")
    payload = {
        "measured": datetime.now().isoformat(timespec="seconds"),
        "frames_averaged": frames,
        "markers_used": used_ids,
        "reprojection_error_pixels": error_pixels,
        "pose": averaged,
        "frame_to_frame_spread": spreads,
        "probe_colour_height_mm": cfg.PROBE_COLOUR_HEIGHT_MM,
    }
    with open(path, "w") as file:
        json.dump(payload, file, indent=2, default=float)
    return path


# ======================================================================
# Shared setup
# ======================================================================

def prepare():
    """Load the calibration and build the detector and board model."""
    try:
        camera_matrix, dist_coeffs, calib_size = load_intrinsics(
            cfg.INTRINSICS_PATH)
    except FileNotFoundError:
        print(f"No calibration file at {cfg.INTRINSICS_PATH}.")
        print("Run capture_calibration_images.py then calibrate_camera.py "
              "first.")
        print()
        print("Without a calibration this tool cannot run. The lens bends")
        print("straight lines, and until we know by how much, an angle")
        print("measured from the markers would be wrong - worst at the edges")
        print("of the frame, which is exactly where the markers sit.")
        return None

    dictionary = cv2.aruco.getPredefinedDictionary(cfg.MARKER_DICTIONARY)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = cfg.ARUCO_THRESH_WIN_MIN
    parameters.adaptiveThreshWinSizeMax = cfg.ARUCO_THRESH_WIN_MAX
    parameters.adaptiveThreshWinSizeStep = cfg.ARUCO_THRESH_WIN_STEP
    # Refine each detected corner to sub-pixel accuracy. This costs a little
    # time and buys a noticeable amount of angular precision, which is the
    # whole point of this tool.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    object_points_by_id = cp.build_marker_object_points(cfg.MARKER_LAYOUT)

    if cfg.POSE_REFERENCE_POINT_MM is None:
        reference = cp.marker_centroid(cfg.MARKER_LAYOUT)
    else:
        reference = cfg.POSE_REFERENCE_POINT_MM

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "calibration_size": calib_size,
        "detector": detector,
        "object_points_by_id": object_points_by_id,
        "reference_point": reference,
    }


def solve_frame(frame, setup):
    """Detect the markers in one frame and solve the pose.
    Returns (rvec, tvec, error, used_ids, corners, ids) or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = setup["detector"].detectMarkers(gray)

    object_points, image_points, used_ids = cp.collect_correspondences(
        corners, ids, setup["object_points_by_id"])
    if not used_ids:
        return None, corners, ids

    solution = cp.solve_pose(object_points, image_points,
                             setup["camera_matrix"], setup["dist_coeffs"])
    if solution is None:
        return None, corners, ids

    rvec, tvec, error = solution
    return (rvec, tvec, error, used_ids), corners, ids


def draw_text(image, text, position, color, scale=0.6):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


# ======================================================================
# Marker-rotation search mode
# ======================================================================

def run_rotation_search(frame, setup):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = setup["detector"].detectMarkers(gray)
    if ids is None or len(ids) < 3:
        found = 0 if ids is None else len(ids)
        print(f"Only {found} markers visible. The search needs at least 3, "
              f"and works best with all 4.")
        return

    print("Trying all 256 rotation combinations...")
    result = cp.solve_marker_rotations(corners, ids, cfg.MARKER_LAYOUT,
                                       setup["camera_matrix"],
                                       setup["dist_coeffs"])
    if result is None:
        print("The search found no workable combination. Check that the marker "
              "IDs and positions in MARKER_LAYOUT match the physical board.")
        return

    print()
    print("BEST FIT")
    for marker_id, rotation in sorted(result["rotations"].items()):
        print(f"    marker {marker_id}: rotation_deg = {rotation}")
    print(f"  Reprojection error : {result['error_pixels']:.3f} px")
    print(f"  Next best combo    : {result['runner_up_error_pixels']:.3f} px")
    print()

    if result["decisive"] and result["error_pixels"] < 2.0:
        print("  This is a clear win. Copy those rotation_deg values into")
        print("  MARKER_LAYOUT in camera_config.py.")
    elif not result["decisive"]:
        print("  WARNING: the best and second-best fits are close together, so")
        print("  the search has not really decided anything. That usually means")
        print("  the marker CENTRE POSITIONS in MARKER_LAYOUT are wrong, and no")
        print("  rotation can rescue them. Re-measure the board.")
    else:
        print("  WARNING: even the best fit has a large reprojection error.")
        print("  Re-measure the marker centre positions and the marker size.")


# ======================================================================
# Main
# ======================================================================

def run_on_image(path, setup):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read {path}")
        return

    calib_width, calib_height = setup["calibration_size"]
    if (frame.shape[1], frame.shape[0]) != (calib_width, calib_height):
        print(f"WARNING: this image is {frame.shape[1]} x {frame.shape[0]} but")
        print(f"         the calibration was made at {calib_width} x "
              f"{calib_height}.")
        print("         The result will be wrong. Use a matching image.")

    solved, corners, ids = solve_frame(frame, setup)
    if solved is None:
        found = 0 if ids is None else len(ids)
        print(f"Could not solve a pose. Markers detected: {found}. "
              f"At least 2 of the layout's markers must be visible.")
        return

    rvec, tvec, error, used_ids = solved
    pose = cp.describe_pose(rvec, tvec, setup["reference_point"])

    print()
    for line in cp.format_pose_report(pose, used_ids, error):
        print(line)
    print_parallax_table(rvec, tvec, cfg.PROBE_COLOUR_HEIGHT_MM,
                         cfg.PARALLAX_SAMPLE_POINTS_MM)


def run_live(setup, solve_rotations_first):
    camera = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open camera at index {cfg.CAMERA_INDEX}.")
        return
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAPTURE_HEIGHT)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    calib_width, calib_height = setup["calibration_size"]
    if (actual_width, actual_height) != (calib_width, calib_height):
        print(f"WARNING: camera is running at {actual_width} x {actual_height} "
              f"but the calibration was made at {calib_width} x {calib_height}.")
        print("         Every distance and angle below will be wrong. Fix the")
        print("         resolution or recalibrate before trusting anything.")

    WINDOW = "Camera pose"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT)

    print()
    print("m measure   p parallax   s save   a solve rotations   q quit")
    print()

    axis_length = max(50.0, 0.25 * max(
        abs(spec["center_mm"][0]) + abs(spec["center_mm"][1])
        for spec in cfg.MARKER_LAYOUT.values()))

    last_measurement = None
    last_rvec = None
    last_tvec = None
    did_rotation_search = not solve_rotations_first

    while True:
        success, frame = camera.read()
        if not success:
            print("Lost the camera. Stopping.")
            break

        if not did_rotation_search:
            run_rotation_search(frame, setup)
            did_rotation_search = True

        solved, corners, ids = solve_frame(frame, setup)

        display = frame.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)

        if solved is not None:
            rvec, tvec, error, used_ids = solved
            last_rvec, last_tvec = rvec, tvec
            cp.draw_board_axes(display, rvec, tvec, setup["camera_matrix"],
                               setup["dist_coeffs"], axis_length)
            pose = cp.describe_pose(rvec, tvec, setup["reference_point"])
            azimuth_text = ("overhead" if pose["azimuth_deg"] is None
                            else f"{pose['azimuth_deg']:7.2f}")
            live_lines = [
                f"markers {used_ids}   error {error:.2f} px",
                f"tilt {pose['tilt_deg']:6.2f}   "
                f"azimuth {azimuth_text}   "
                f"roll {pose['image_roll_deg']:7.2f}",
                f"position  X {pose['camera_x_mm']:7.1f}  "
                f"Y {pose['camera_y_mm']:7.1f}  Z {pose['camera_z_mm']:7.1f} mm",
            ]
            color = (0, 255, 0) if error < 2.0 else (0, 165, 255)
        else:
            found = 0 if ids is None else len(ids)
            live_lines = [f"No pose. Markers seen: {found}",
                          "Need at least 2 of the layout's markers in view."]
            color = (0, 0, 255)

        display = cv2.resize(display, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))
        for i, line in enumerate(live_lines):
            draw_text(display, line, (20, 40 + i * 30), color)
        draw_text(display, "m measure   p parallax   s save   a rotations   q quit",
                  (20, cfg.DISPLAY_HEIGHT - 20), (200, 200, 200), scale=0.55)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('m'):
            print(f"Measuring over {MEASUREMENT_FRAMES} frames. "
                  f"Do not touch anything...")
            poses = []
            errors = []
            ids_seen = []
            for _ in range(MEASUREMENT_FRAMES):
                ok, measure_frame = camera.read()
                if not ok:
                    continue
                solved, _, _ = solve_frame(measure_frame, setup)
                if solved is None:
                    continue
                rvec, tvec, error, used_ids = solved
                poses.append(cp.describe_pose(rvec, tvec,
                                              setup["reference_point"]))
                errors.append(error)
                ids_seen = used_ids
                last_rvec, last_tvec = rvec, tvec

            if len(poses) < MEASUREMENT_FRAMES // 2:
                print(f"Only {len(poses)} frames solved. The markers are not "
                      f"being seen reliably; not reporting a measurement.")
                continue

            averaged, spreads = average_poses(poses)
            mean_error = float(np.mean(errors))
            print_measurement(averaged, spreads, ids_seen, mean_error,
                              len(poses))
            last_measurement = (averaged, spreads, ids_seen, mean_error,
                                len(poses))

        elif key == ord('p'):
            if last_rvec is None:
                print("No pose yet - point the camera at the markers first.")
            else:
                print_parallax_table(last_rvec, last_tvec,
                                     cfg.PROBE_COLOUR_HEIGHT_MM,
                                     cfg.PARALLAX_SAMPLE_POINTS_MM)

        elif key == ord('s'):
            if last_measurement is None:
                print("Nothing measured yet. Press 'm' first.")
            else:
                path = save_measurement(cfg.POSE_OUTPUT_FOLDER,
                                        *last_measurement)
                print(f"Saved {path}")

        elif key == ord('a'):
            run_rotation_search(frame, setup)

    camera.release()
    cv2.destroyAllWindows()
    print("Stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Measure the camera pose from the four board ArUco markers.")
    parser.add_argument("--image", help="measure from a saved image instead of "
                                        "the live camera")
    parser.add_argument("--solve-rotations", action="store_true",
                        help="run the marker-rotation search on the first frame")
    args = parser.parse_args()

    setup = prepare()
    if setup is None:
        return

    if args.image:
        if args.solve_rotations:
            frame = cv2.imread(args.image)
            if frame is None:
                print(f"Could not read {args.image}")
                return
            run_rotation_search(frame, setup)
        else:
            run_on_image(args.image, setup)
    else:
        run_live(setup, args.solve_rotations)


if __name__ == "__main__":
    main()
