# capture_calibration_images.py
#
# Interactive tool for photographing the printed ChArUco board.
#
#     python capture_calibration_images.py
#
# WHY THIS TOOL EXISTS RATHER THAN "just take some photos"
#
# Calibration works out several unknowns at once: the focal length, where the
# optical centre of the lens sits on the sensor, and how the lens bends
# straight lines. Each of those unknowns needs a different kind of evidence:
#
#   - Focal length needs the board at DIFFERENT DISTANCES.
#   - The optical centre and the distortion terms need the board in DIFFERENT
#     PARTS OF THE FRAME - especially the corners, because that is where lens
#     distortion is strongest. A pile of photographs all taken with the board
#     centred will produce a confident-looking calibration whose distortion
#     numbers are close to meaningless.
#   - Everything needs the board at DIFFERENT ANGLES. A set of photographs all
#     taken flat-on is mathematically ambiguous; the solver cannot separate
#     "the board is far away" from "the focal length is long".
#
# So this tool shows you a live coverage grid: the frame is divided into cells,
# and a cell lights up once you have captured board corners inside it. Your job
# is to light up every cell, including the four corner cells.
#
# It also refuses blurry frames. Motion blur moves chessboard corners by a
# fraction of a pixel in a direction that depends on how you were moving, which
# is exactly the kind of error calibration cannot see and cannot correct.
#
# CONTROLS
#     SPACE  save the current frame (only if it is sharp and has enough corners)
#     u      delete the most recently saved frame
#     c      clear the coverage grid
#     q      finish
#
# HOW TO MOVE THE BOARD
#     Hold the board, not the camera. The camera must stay exactly where the
#     real system uses it. Work through: close / mid / far, each at flat-on and
#     tilted roughly 30 degrees four different ways, and each in the centre and
#     in all four corners of the frame. Tilt the BOARD, do not just slide it.

import os

import cv2
import numpy as np

import camera_config as cfg

# The frame is divided into this many cells for the coverage display.
COVERAGE_COLUMNS = 6
COVERAGE_ROWS = 4

# A frame is rejected as blurry if this "sharpness score" is below the
# threshold. The score is the variance of the Laplacian: a standard, cheap
# measure of how much fine detail an image contains. A sharp chessboard is full
# of hard edges and scores high; a blurred one scores low.
#
# The right threshold depends on your lighting and lens, so the tool prints the
# live score. Point the camera at the sharp, well-lit board, note the score,
# then set this to roughly half of it.
BLUR_THRESHOLD = 60.0


def compute_sharpness(gray_image):
    """Return a 'how much fine detail is here' score. Higher is sharper.

    We shrink the image first. At 4K the Laplacian is slow enough to make the
    live view stutter, and blur is a large-scale effect that survives being
    scaled down, so nothing useful is lost."""
    small = cv2.resize(gray_image, (0, 0), fx=0.25, fy=0.25)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def mark_coverage(coverage_grid, corners, frame_width, frame_height):
    """Record which cells of the frame these detected corners fall into."""
    for corner in corners.reshape(-1, 2):
        column = int(corner[0] * COVERAGE_COLUMNS / frame_width)
        row = int(corner[1] * COVERAGE_ROWS / frame_height)
        column = max(0, min(COVERAGE_COLUMNS - 1, column))
        row = max(0, min(COVERAGE_ROWS - 1, row))
        coverage_grid[row][column] += 1


def draw_coverage(display_image, coverage_grid):
    """Overlay the coverage grid. Filled green = covered, hollow red = not."""
    height, width = display_image.shape[:2]
    cell_width = width / COVERAGE_COLUMNS
    cell_height = height / COVERAGE_ROWS

    overlay = display_image.copy()
    for row in range(COVERAGE_ROWS):
        for column in range(COVERAGE_COLUMNS):
            x1 = int(column * cell_width)
            y1 = int(row * cell_height)
            x2 = int((column + 1) * cell_width)
            y2 = int((row + 1) * cell_height)
            if coverage_grid[row][column] > 0:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), -1)
            else:
                cv2.rectangle(overlay, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2),
                              (0, 0, 220), 2)
    cv2.addWeighted(overlay, 0.22, display_image, 0.78, 0, display_image)

    # Grid lines on top so the cells stay readable.
    for column in range(1, COVERAGE_COLUMNS):
        x = int(column * cell_width)
        cv2.line(display_image, (x, 0), (x, height), (90, 90, 90), 1)
    for row in range(1, COVERAGE_ROWS):
        y = int(row * cell_height)
        cv2.line(display_image, (0, y), (width, y), (90, 90, 90), 1)


def draw_text(image, text, position, color, scale=0.7):
    """Bright text with a black outline, readable over anything."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def main():
    os.makedirs(cfg.CALIBRATION_IMAGES_FOLDER, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(cfg.CHARUCO_DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (cfg.CHARUCO_SQUARES_X, cfg.CHARUCO_SQUARES_Y),
        cfg.CHARUCO_SQUARE_LENGTH_MM,
        cfg.CHARUCO_MARKER_LENGTH_MM,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)

    camera = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open camera at index {cfg.CAMERA_INDEX}.")
        return
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAPTURE_HEIGHT)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera open at {actual_width} x {actual_height}")
    if (actual_width, actual_height) != (cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGHT):
        print("WARNING: the camera did not accept the requested resolution.")
        print("         Calibration is only valid at the resolution you")
        print("         actually run the system at. Fix this before capturing.")
    print()
    print("SPACE save  |  u undo last  |  c clear coverage  |  q finish")
    print()

    WINDOW = "Calibration capture"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT)

    coverage_grid = [[0] * COVERAGE_COLUMNS for _ in range(COVERAGE_ROWS)]
    saved_paths = []

    # Start the numbering after any images already in the folder, so a second
    # session adds to the set instead of overwriting the first one.
    existing = [f for f in os.listdir(cfg.CALIBRATION_IMAGES_FOLDER)
                if f.lower().endswith(".png")]
    next_index = len(existing)
    if existing:
        print(f"Found {len(existing)} images already in "
              f"{cfg.CALIBRATION_IMAGES_FOLDER}; adding to them.")

    while True:
        success, frame = camera.read()
        if not success:
            print("Lost the camera. Stopping.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = compute_sharpness(gray)

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            detector.detectBoard(gray)

        corner_count = 0 if charuco_ids is None else len(charuco_ids)
        sharp_enough = sharpness >= BLUR_THRESHOLD
        enough_corners = corner_count >= cfg.MIN_CHARUCO_CORNERS
        can_save = sharp_enough and enough_corners

        # --- Build the preview -------------------------------------------
        display = cv2.resize(frame, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))
        draw_coverage(display, coverage_grid)

        if charuco_corners is not None and len(charuco_corners) > 0:
            scale_x = cfg.DISPLAY_WIDTH / frame.shape[1]
            scale_y = cfg.DISPLAY_HEIGHT / frame.shape[0]
            for corner in charuco_corners.reshape(-1, 2):
                point = (int(corner[0] * scale_x), int(corner[1] * scale_y))
                cv2.circle(display, point, 4, (0, 255, 255), -1)

        covered_cells = sum(1 for row in coverage_grid for cell in row if cell > 0)
        total_cells = COVERAGE_COLUMNS * COVERAGE_ROWS

        status_color = (0, 255, 0) if can_save else (0, 165, 255)
        draw_text(display, f"Corners: {corner_count} "
                           f"(need {cfg.MIN_CHARUCO_CORNERS})",
                  (20, 40), (0, 255, 0) if enough_corners else (0, 165, 255))
        draw_text(display, f"Sharpness: {sharpness:6.1f} "
                           f"(need {BLUR_THRESHOLD:.0f})",
                  (20, 72), (0, 255, 0) if sharp_enough else (0, 0, 255))
        draw_text(display, f"Saved: {len(saved_paths) + len(existing)} "
                           f"(target {cfg.TARGET_CALIBRATION_IMAGES})",
                  (20, 104), (255, 255, 255))
        draw_text(display, f"Frame coverage: {covered_cells}/{total_cells} cells",
                  (20, 136), (0, 255, 0) if covered_cells == total_cells
                  else (0, 165, 255))
        draw_text(display, "SPACE save   u undo   c clear   q finish",
                  (20, cfg.DISPLAY_HEIGHT - 24), (200, 200, 200), scale=0.6)

        if not can_save:
            reason = "TOO BLURRY" if not sharp_enough else "NOT ENOUGH BOARD"
            draw_text(display, reason, (cfg.DISPLAY_WIDTH // 2 - 110, 40),
                      (0, 0, 255), scale=0.9)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        # --- Keys ---------------------------------------------------------
        if key == ord('q'):
            break

        elif key == ord(' '):
            if not can_save:
                print("Not saved: " +
                      ("too blurry. " if not sharp_enough else "") +
                      ("not enough board corners visible." if not enough_corners
                       else ""))
                continue
            path = os.path.join(cfg.CALIBRATION_IMAGES_FOLDER,
                                f"calib_{next_index:03d}.png")
            # PNG, not JPG. JPEG compression softens edges by a fraction of a
            # pixel, and a fraction of a pixel is exactly the accuracy we are
            # trying to keep.
            cv2.imwrite(path, frame)
            saved_paths.append(path)
            next_index += 1
            mark_coverage(coverage_grid, charuco_corners,
                          frame.shape[1], frame.shape[0])
            print(f"Saved {path}   corners={corner_count}  "
                  f"sharpness={sharpness:.1f}")

        elif key == ord('u'):
            if saved_paths:
                path = saved_paths.pop()
                os.remove(path)
                next_index -= 1
                print(f"Deleted {path}")
                print("  (the coverage grid still shows that cell as covered; "
                      "press 'c' to reset it if that matters)")
            else:
                print("Nothing saved this session to undo.")

        elif key == ord('c'):
            coverage_grid = [[0] * COVERAGE_COLUMNS for _ in range(COVERAGE_ROWS)]
            print("Coverage grid cleared.")

    camera.release()
    cv2.destroyAllWindows()

    total = len(saved_paths) + len(existing)
    print()
    print(f"Finished with {total} calibration images in "
          f"{cfg.CALIBRATION_IMAGES_FOLDER}")
    if total < cfg.TARGET_CALIBRATION_IMAGES:
        print(f"That is fewer than the target of "
              f"{cfg.TARGET_CALIBRATION_IMAGES}. The calibration will still "
              f"run, but expect it to be less stable.")
    print("Next: python calibrate_camera.py")


if __name__ == "__main__":
    main()
