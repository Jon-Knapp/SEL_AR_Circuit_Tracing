# test_pose_math.py
#
# A self-check for camera_pose.py that needs NO camera and NO printed markers.
#
#     python test_pose_math.py
#
# HOW IT WORKS
#   We invent a camera at a pose we choose ourselves - say, 30 degrees off
#   vertical, 45 degrees round, 900 mm away - and use the standard projection
#   maths to work out exactly which pixel each marker corner would land on.
#   Then we hand those pixels to camera_pose.py and check it gives us back the
#   pose we started with.
#
#   This is worth doing before you touch hardware. If this test fails, the
#   problem is in the code. If it passes and the real measurements still look
#   wrong, the problem is in the physical measurements you typed into
#   camera_config.py - and knowing which of those two it is saves hours.

import math

import cv2
import numpy as np

import camera_pose as cp

# A pretend board: four 40 mm markers on a 700 x 500 mm rectangle.
TEST_LAYOUT = {
    0: {"center_mm": (0.0,   0.0),   "size_mm": 40.0, "rotation_deg": 0},
    3: {"center_mm": (700.0, 0.0),   "size_mm": 40.0, "rotation_deg": 0},
    4: {"center_mm": (0.0,   500.0), "size_mm": 40.0, "rotation_deg": 0},
    5: {"center_mm": (700.0, 500.0), "size_mm": 40.0, "rotation_deg": 0},
}

# A pretend 4K camera with a roughly 90 degree lens and no distortion.
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
FOCAL_LENGTH = 1920.0
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0.0, IMAGE_WIDTH / 2.0],
    [0.0, FOCAL_LENGTH, IMAGE_HEIGHT / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)


def make_known_pose(reference_point, tilt_deg, azimuth_deg, distance_mm):
    """
    Build the rvec / tvec for a camera sitting at a chosen tilt, azimuth and
    distance from a reference point, aimed straight at that point.

    This is the reverse of what describe_pose does, so running one after the
    other should get us back where we started.
    """
    tilt = math.radians(tilt_deg)
    azimuth = math.radians(azimuth_deg)

    reference = np.array([reference_point[0], reference_point[1], 0.0])
    direction_to_camera = np.array([
        math.sin(tilt) * math.cos(azimuth),
        math.sin(tilt) * math.sin(azimuth),
        math.cos(tilt),
    ])
    camera_position = reference + distance_mm * direction_to_camera

    # The camera looks from its position toward the reference point.
    forward = reference - camera_position
    forward = forward / np.linalg.norm(forward)

    # Pick any direction that is not parallel to 'forward' to define which way
    # is up, then build a proper right-handed camera frame from it.
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up_hint, forward))) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])     # camera is straight down
    right = np.cross(up_hint, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    # Rows of the rotation matrix are the camera's axes in board coordinates.
    rotation = np.array([right, down, forward])
    translation = -rotation @ camera_position

    rvec, _ = cv2.Rodrigues(rotation)
    tvec = translation.reshape(3, 1)
    return rvec, tvec, camera_position


def project_markers(layout, rvec, tvec):
    """Work out where each marker's corners would land in the image, and
    package them the way cv2.aruco.detectMarkers would."""
    object_points_by_id = cp.build_marker_object_points(layout)
    corners_list = []
    ids_list = []
    for marker_id in sorted(object_points_by_id):
        object_points = object_points_by_id[marker_id]
        projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                         CAMERA_MATRIX, DIST_COEFFS)
        corners_list.append(projected.reshape(1, 4, 2).astype(np.float32))
        ids_list.append([marker_id])
    return corners_list, np.array(ids_list, dtype=np.int32)


def check(label, actual, expected, tolerance, unit=""):
    """Compare a measured value against the expected one.

    When the value is an ANGLE we compare the two the short way round the
    circle, because -180 degrees and +180 degrees are the same direction even
    though plain subtraction says they are 360 apart. Getting this wrong in a
    test is a classic way to chase a bug that is not there."""
    if unit.strip() == "deg":
        difference = abs((actual - expected + 180.0) % 360.0 - 180.0)
    else:
        difference = abs(actual - expected)
    passed = difference <= tolerance
    mark = "PASS" if passed else "FAIL"
    print(f"    [{mark}] {label:<26} got {actual:9.3f}{unit}  "
          f"expected {expected:9.3f}{unit}  (off by {difference:.4f})")
    return passed


def run_case(tilt_deg, azimuth_deg, distance_mm):
    reference = cp.marker_centroid(TEST_LAYOUT)
    rvec, tvec, true_position = make_known_pose(reference, tilt_deg,
                                                azimuth_deg, distance_mm)
    corners, ids = project_markers(TEST_LAYOUT, rvec, tvec)

    object_points_by_id = cp.build_marker_object_points(TEST_LAYOUT)
    object_points, image_points, used = cp.collect_correspondences(
        corners, ids, object_points_by_id)

    solution = cp.solve_pose(object_points, image_points,
                             CAMERA_MATRIX, DIST_COEFFS)
    if solution is None:
        print("    [FAIL] solve_pose returned nothing")
        return False

    recovered_rvec, recovered_tvec, error = solution
    pose = cp.describe_pose(recovered_rvec, recovered_tvec, reference)

    print(f"  Case: tilt {tilt_deg} deg, azimuth {azimuth_deg} deg, "
          f"distance {distance_mm} mm")
    print(f"    reprojection error {error:.5f} px, markers {used}")

    results = [
        check("tilt", pose["tilt_deg"], tilt_deg, 0.05, " deg"),
        check("distance", pose["distance_mm"], distance_mm, 0.5, " mm"),
        check("camera X", pose["camera_x_mm"], true_position[0], 0.5, " mm"),
        check("camera Y", pose["camera_y_mm"], true_position[1], 0.5, " mm"),
        check("camera Z", pose["camera_z_mm"], true_position[2], 0.5, " mm"),
        # The camera is aimed at the reference point, so its aim tilt should
        # equal its position tilt.
        check("aim tilt", pose["aim_tilt_deg"], tilt_deg, 0.05, " deg"),
    ]

    # Azimuth is only a real quantity once the camera is off vertical. At tilt
    # zero the correct answer is "undefined", so we check for that instead of
    # checking for a number.
    if tilt_deg < 0.01:
        reported_none = pose["azimuth_deg"] is None
        mark = "PASS" if reported_none else "FAIL"
        print(f"    [{mark}] {'azimuth':<26} correctly reported as "
              f"{'undefined' if reported_none else pose['azimuth_deg']} "
              f"(camera is straight overhead)")
        results.append(reported_none)
    else:
        results.append(
            check("azimuth", pose["azimuth_deg"], azimuth_deg, 0.05, " deg"))
    if pose["gimbal_lock_warning"]:
        print("    (gimbal lock warning raised, as expected near vertical)")
    print()
    return all(results)


def run_parallax_checks():
    """Check the parallax prediction against cases we can work out by hand."""
    print("  Parallax checks")
    reference = cp.marker_centroid(TEST_LAYOUT)
    height = 12.0
    all_passed = True

    # Case 1: camera directly above the reference point, probe AT that point.
    # There is no sideways offset between camera and probe, so the raised
    # colour sits exactly on top of the tip and the error must be zero.
    rvec, tvec, _ = make_known_pose(reference, 0.0, 0.0, 900.0)
    result = cp.parallax_offset(rvec, tvec, reference, height)
    all_passed &= check("overhead, centre", result[2], 0.0, 0.001, " mm")

    # Case 2: camera directly above the reference point, probe 300 mm to one
    # side. Similar triangles: the offset is height x (sideways distance) /
    # (camera height above the colour).
    offset_point = (reference[0] + 300.0, reference[1])
    result = cp.parallax_offset(rvec, tvec, offset_point, height)
    expected = height * 300.0 / (900.0 - height)
    all_passed &= check("overhead, 300 mm out", result[2], expected, 0.01, " mm")

    # Case 3: a very distant camera at 30 degrees tilt, probe at the reference
    # point. With the camera far away the rays are almost parallel, so the
    # simple height x tan(tilt) formula from the test plan should be very close.
    rvec, tvec, _ = make_known_pose(reference, 30.0, 0.0, 100000.0)
    result = cp.parallax_offset(rvec, tvec, reference, height)
    expected = height * math.tan(math.radians(30.0))
    all_passed &= check("far camera, 30 deg", result[2], expected, 0.01, " mm")
    print()
    return all_passed


def run_rotation_search_check():
    """Scramble the marker rotations, then check the search finds them."""
    print("  Marker-rotation search")
    truth = {0: 90, 3: 180, 4: 0, 5: 270}
    scrambled = {}
    for marker_id, rotation in truth.items():
        spec = dict(TEST_LAYOUT[marker_id])
        spec["rotation_deg"] = rotation
        scrambled[marker_id] = spec

    reference = cp.marker_centroid(TEST_LAYOUT)
    rvec, tvec, _ = make_known_pose(reference, 25.0, 60.0, 850.0)
    corners, ids = project_markers(scrambled, rvec, tvec)

    # Hand the search a layout that says every rotation is 0 - i.e. the state
    # you would be in before measuring them - and see if it recovers the truth.
    result = cp.solve_marker_rotations(corners, ids, TEST_LAYOUT,
                                       CAMERA_MATRIX, DIST_COEFFS)
    found = result["rotations"]
    passed = found == truth
    mark = "PASS" if passed else "FAIL"
    print(f"    [{mark}] recovered {found}")
    print(f"           expected  {truth}")
    print(f"           best error {result['error_pixels']:.4f} px, "
          f"runner-up {result['runner_up_error_pixels']:.4f} px, "
          f"decisive={result['decisive']}")
    print()
    return passed


def main():
    print()
    print("=" * 70)
    print("SELF-CHECK FOR camera_pose.py")
    print("=" * 70)
    print()
    print("  Pose recovery")
    print()

    all_passed = True
    for tilt, azimuth, distance in [
        (0.0,  0.0,   900.0),      # the overhead baseline (gimbal lock case)
        (15.0, 0.0,   900.0),
        (30.0, 45.0,  900.0),
        (30.0, 180.0, 750.0),
        (45.0, -90.0, 1100.0),
        (60.0, 135.0, 800.0),
    ]:
        all_passed &= run_case(tilt, azimuth, distance)

    all_passed &= run_parallax_checks()
    all_passed &= run_rotation_search_check()

    print("=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED - the maths in camera_pose.py is sound.")
        print("If real measurements still look wrong, the problem is in the")
        print("numbers you measured and typed into camera_config.py.")
    else:
        print("SOME CHECKS FAILED - do not trust measurements from this code")
        print("until they pass.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
