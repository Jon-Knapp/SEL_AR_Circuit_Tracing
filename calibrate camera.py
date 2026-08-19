# calibrate_camera.py
#
# Turns the folder of ChArUco photographs into the camera's intrinsic
# calibration, and writes it to camera_intrinsics.json.
#
#     python calibrate_camera.py
#
# WHAT "INTRINSIC CALIBRATION" MEANS, IN PLAIN TERMS
#
# A camera turns 3D points in the world into 2D points in an image. To undo
# that - which is what pose estimation does - you have to know the rules the
# camera used. There are two groups of rules:
#
#   INTRINSICS (this script): properties of the camera itself, which do not
#   change when you move it.
#       fx, fy   the focal length, in PIXELS. Roughly "how zoomed in am I".
#       cx, cy   the principal point: where the optical axis actually crosses
#                the sensor. Usually near the middle of the image, but not
#                exactly, because lenses are not glued on perfectly.
#       k1,k2,k3 radial distortion: how much the lens bows straight lines
#                outward or inward. Wide lenses like the Facecam's ~90 degree
#                lens have plenty of this.
#       p1,p2    tangential distortion: the small effect of the lens not being
#                perfectly parallel to the sensor.
#
#   EXTRINSICS (the other script, measure_camera_pose.py): where the camera is
#   and which way it points. These DO change when you move it.
#
# You must know the intrinsics before you can solve for the extrinsics. That is
# the whole reason this script exists: without it, an angle measured from the
# ArUco markers would be contaminated by lens distortion, and the error would
# be worst at the edges of the frame - exactly where you care most.
#
# HOW THE MATHS WORKS, IN ONE PARAGRAPH
#
# Each photograph gives us a set of pairs: "this chessboard corner is at this
# known place on the printed board (in mm), and it landed at this pixel". For a
# guessed set of intrinsics and a guessed board position, we can PREDICT where
# each corner should land. The gap between predicted and actual is the
# reprojection error. The solver adjusts the intrinsics, and each photograph's
# board position, until the total error is as small as it can be. That is all
# calibration is: a big least-squares fit, measured in pixels.

import json
import os
from datetime import datetime

import cv2
import numpy as np

import camera_config as cfg


def build_board():
    """Create the ChArUco board definition described in camera_config.py."""
    dictionary = cv2.aruco.getPredefinedDictionary(cfg.CHARUCO_DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (cfg.CHARUCO_SQUARES_X, cfg.CHARUCO_SQUARES_Y),
        cfg.CHARUCO_SQUARE_LENGTH_MM,
        cfg.CHARUCO_MARKER_LENGTH_MM,
        dictionary,
    )
    return board


def collect_observations(board, image_paths):
    """
    Look at every photograph and pull out the pairs of
    (known board position in mm, observed pixel position).

    Returns three parallel lists:
        object_point_sets : one (N, 1, 3) array per usable photograph
        image_point_sets  : one (N, 1, 2) array per usable photograph
        used_paths        : the file each set came from
    plus the image size, and a list of (path, reason) for the rejects.
    """
    detector = cv2.aruco.CharucoDetector(board)

    object_point_sets = []
    image_point_sets = []
    used_paths = []
    rejected = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            rejected.append((path, "could not be read"))
            continue

        if image_size is None:
            image_size = (image.shape[1], image.shape[0])   # (width, height)
        elif (image.shape[1], image.shape[0]) != image_size:
            # Mixing resolutions would silently corrupt the result, because
            # the intrinsics are expressed in pixels of one specific size.
            rejected.append((path, "different resolution from the others"))
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < cfg.MIN_CHARUCO_CORNERS:
            found = 0 if charuco_ids is None else len(charuco_ids)
            rejected.append((path, f"only {found} corners found "
                                   f"(need {cfg.MIN_CHARUCO_CORNERS})"))
            continue

        # matchImagePoints pairs each detected chessboard corner with its known
        # physical position on the printed board. This is the modern OpenCV way
        # and it replaces the old calibrateCameraCharuco helper.
        object_points, image_points = board.matchImagePoints(
            charuco_corners, charuco_ids)

        if object_points is None or len(object_points) < 4:
            rejected.append((path, "corners could not be matched to the board"))
            continue

        object_point_sets.append(object_points)
        image_point_sets.append(image_points)
        used_paths.append(path)

    return (object_point_sets, image_point_sets, used_paths, image_size,
            rejected)


def per_view_errors(object_point_sets, image_point_sets, rvecs, tvecs,
                    camera_matrix, dist_coeffs):
    """
    For each photograph, work out its own average reprojection error in pixels.

    This matters because the overall error is an average, and one badly blurred
    or slightly bent photograph can drag the whole calibration off while the
    average still looks acceptable. Finding and removing the outliers usually
    improves the result more than adding more photographs does.
    """
    errors = []
    for i in range(len(object_point_sets)):
        projected, _ = cv2.projectPoints(object_point_sets[i], rvecs[i],
                                         tvecs[i], camera_matrix, dist_coeffs)
        difference = image_point_sets[i].reshape(-1, 2) - projected.reshape(-1, 2)
        # Root-mean-square distance, in pixels.
        error = float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))
        errors.append(error)
    return errors


def run_calibration(object_point_sets, image_point_sets, image_size):
    """Do the least-squares fit. Returns the OpenCV calibration outputs."""
    flags = 0
    if cfg.USE_RATIONAL_DISTORTION_MODEL:
        flags |= cv2.CALIB_RATIONAL_MODEL

    return cv2.calibrateCamera(
        object_point_sets, image_point_sets, image_size,
        cameraMatrix=None, distCoeffs=None, flags=flags)


def save_intrinsics(path, camera_matrix, dist_coeffs, image_size,
                    overall_error, image_count):
    """Write the calibration to JSON so the pose tool can read it back."""
    data = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.flatten().tolist(),
        "overall_reprojection_error_pixels": float(overall_error),
        "images_used": int(image_count),
        "charuco_squares_x": cfg.CHARUCO_SQUARES_X,
        "charuco_squares_y": cfg.CHARUCO_SQUARES_Y,
        "charuco_square_length_mm": cfg.CHARUCO_SQUARE_LENGTH_MM,
        "charuco_marker_length_mm": cfg.CHARUCO_MARKER_LENGTH_MM,
        "rational_model": bool(cfg.USE_RATIONAL_DISTORTION_MODEL),
        "note": ("Intrinsics are valid ONLY at the image size recorded above, "
                 "and only for the camera settings in force when the "
                 "photographs were taken. Re-calibrate if either changes."),
    }
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_intrinsics(path):
    """Read a saved calibration back. Returns (camera_matrix, dist_coeffs,
    (width, height)) or raises FileNotFoundError."""
    with open(path) as file:
        data = json.load(file)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float64)
    image_size = (data["image_width"], data["image_height"])
    return camera_matrix, dist_coeffs, image_size


def describe_result(camera_matrix, dist_coeffs, image_size, overall_error):
    """Print the calibration in human terms, with sanity checks."""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    width, height = image_size

    print()
    print("=" * 66)
    print("CALIBRATION RESULT")
    print("=" * 66)
    print(f"  Image size          : {width} x {height} px")
    print(f"  Focal length fx, fy : {fx:.2f}, {fy:.2f} px")
    print(f"  Principal point     : ({cx:.2f}, {cy:.2f}) px")
    print(f"  Distortion          : "
          f"{', '.join(f'{c:.5f}' for c in dist_coeffs.flatten())}")
    print(f"  Reprojection error  : {overall_error:.4f} px")
    print()

    # --- Sanity check 1: field of view.
    # Convert the focal length into an angle so it can be compared against the
    # lens specification. If these disagree badly, something is wrong.
    horizontal_fov = 2 * np.degrees(np.arctan(width / (2 * fx)))
    vertical_fov = 2 * np.degrees(np.arctan(height / (2 * fy)))
    diagonal = np.hypot(width, height)
    diagonal_fov = 2 * np.degrees(np.arctan(diagonal / (2 * fx)))
    print(f"  Implied field of view: {horizontal_fov:.1f} deg horizontal, "
          f"{vertical_fov:.1f} deg vertical, {diagonal_fov:.1f} deg diagonal")
    print("    Compare this against the camera's published field of view. The")
    print("    Facecam 4K is quoted at about 90 degrees. If your number is far")
    print("    off, check that the square size in camera_config.py is the")
    print("    MEASURED printed size, not the intended one.")
    print()

    # --- Sanity check 2: is the principal point plausible?
    offset_x = abs(cx - width / 2) / width
    offset_y = abs(cy - height / 2) / height
    if offset_x > 0.10 or offset_y > 0.10:
        print("  WARNING: the principal point is more than 10% of the frame")
        print("           away from the image centre. That is unusual and")
        print("           normally means the photographs did not cover enough")
        print("           of the frame. Capture more views with the board in")
        print("           the corners, then calibrate again.")
        print()

    # --- Sanity check 3: is fx roughly equal to fy?
    aspect = fx / fy
    if not (0.97 <= aspect <= 1.03):
        print(f"  WARNING: fx / fy = {aspect:.4f}. For a normal camera with")
        print("           square pixels this should be very close to 1.000.")
        print("           A value far from 1 usually means the camera is")
        print("           stretching or cropping the image - check that no")
        print("           digital zoom or aspect correction is switched on in")
        print("           Elgato Camera Hub.")
        print()

    # --- Sanity check 4: the headline number.
    if overall_error < 0.3:
        print("  Reprojection error is excellent. Use this calibration.")
    elif overall_error < 0.5:
        print("  Reprojection error is good. Use this calibration.")
    elif overall_error < 1.0:
        print("  Reprojection error is usable but not great. Consider")
        print("  re-shooting the worst views listed above, or capturing more")
        print("  angles, before relying on this for careful measurements.")
    else:
        print("  Reprojection error is POOR. Do not trust this calibration.")
        print("  Usual causes, in order of likelihood:")
        print("    - the printed board is not flat (bowed paper)")
        print("    - the square size in camera_config.py does not match what")
        print("      was actually printed")
        print("    - blurred photographs")
        print("    - all photographs taken from similar angles")
        print("    - the lens genuinely needs the rational distortion model;")
        print("      try USE_RATIONAL_DISTORTION_MODEL = True")
    print("=" * 66)


def main():
    if not os.path.isdir(cfg.CALIBRATION_IMAGES_FOLDER):
        print(f"No folder '{cfg.CALIBRATION_IMAGES_FOLDER}'.")
        print("Run capture_calibration_images.py first.")
        return

    image_paths = sorted(
        os.path.join(cfg.CALIBRATION_IMAGES_FOLDER, name)
        for name in os.listdir(cfg.CALIBRATION_IMAGES_FOLDER)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_paths:
        print(f"No images found in {cfg.CALIBRATION_IMAGES_FOLDER}.")
        return

    print(f"Found {len(image_paths)} images. Detecting the board in each...")
    board = build_board()
    (object_point_sets, image_point_sets, used_paths, image_size,
     rejected) = collect_observations(board, image_paths)

    for path, reason in rejected:
        print(f"  SKIPPED {os.path.basename(path)}: {reason}")

    print(f"\nUsing {len(used_paths)} of {len(image_paths)} images.")
    if len(used_paths) < 5:
        print("That is not enough to calibrate. Capture more images.")
        return
    if len(used_paths) < cfg.TARGET_CALIBRATION_IMAGES:
        print(f"Fewer than the target of {cfg.TARGET_CALIBRATION_IMAGES}. "
              f"The result will be less stable than it could be.")

    print("Calibrating (this takes a little while at 4K)...")
    try:
        overall_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
            run_calibration(object_point_sets, image_point_sets, image_size)
    except cv2.error as error:
        print("\nOpenCV could not solve the calibration:")
        print(f"   {error}")
        print("\nThe usual cause is that every photograph was taken from a")
        print("similar angle. The solver needs the board TILTED in several")
        print("different directions, not just moved around flat-on.")
        return

    # --- Report which individual views fit badly.
    errors = per_view_errors(object_point_sets, image_point_sets, rvecs, tvecs,
                             camera_matrix, dist_coeffs)
    ranked = sorted(zip(errors, used_paths), reverse=True)

    print("\nWorst-fitting images (delete these and re-run if any stand out):")
    for error, path in ranked[:5]:
        flag = "  <-- suspect" if error > cfg.SUSPECT_VIEW_ERROR_PIXELS else ""
        print(f"   {error:6.3f} px   {os.path.basename(path)}{flag}")

    describe_result(camera_matrix, dist_coeffs, image_size, overall_error)

    save_intrinsics(cfg.INTRINSICS_PATH, camera_matrix, dist_coeffs,
                    image_size, overall_error, len(used_paths))
    print(f"\nWrote {cfg.INTRINSICS_PATH}")
    print("Next: python measure_camera_pose.py")


if __name__ == "__main__":
    main()
