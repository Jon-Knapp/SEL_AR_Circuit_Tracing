# camera_config.py
#
# Every setting for the two new tools:
#
#   1. CAMERA CALIBRATION - working out the camera's internal optics
#      (focal length, lens centre, lens distortion) from photographs of a
#      printed ChArUco board.
#
#   2. CAMERA POSE MEASUREMENT - working out where the camera is and which
#      way it is pointing, relative to the plywood work surface, using the
#      four ArUco markers that are already on that surface.
#
# This is a SEPARATE file from the delivered system's config.py on purpose.
# Nothing here changes how main.py behaves. These are measurement tools.
#
# Coordinate system used throughout (memorise this, everything depends on it):
#
#       Z  (out of the plywood, toward the camera)
#       |
#       |
#       +-------- X  (to the operator's right, along the board)
#      /
#     Y  (away from the operator, across the board)
#
#   All distances are in MILLIMETRES.
#   Z = 0 is the surface of the plywood.
#   The camera therefore always has a POSITIVE Z value.

import cv2

# ======================================================================
# SECTION 1 - Camera
# ======================================================================

# Which camera to open. Must be the same one the real system uses.
CAMERA_INDEX = 1

# Capture resolution.
#
# CRITICAL: calibrate at the EXACT resolution you run main.py at. The camera
# matrix scales with image size, so intrinsics measured at 1920x1080 are wrong
# for 3840x2160. If you change CAPTURE_WIDTH/HEIGHT in the main config.py, you
# must recalibrate.
CAPTURE_WIDTH = 3840
CAPTURE_HEIGHT = 2160

# On-screen preview size for the interactive tools.
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# ======================================================================
# SECTION 2 - The printed ChArUco calibration board
# ======================================================================
#
# A ChArUco board is a chessboard with an ArUco marker printed inside every
# white square. You get the best of both:
#   - chessboard corners can be located to sub-pixel accuracy, which is what
#     makes calibration precise;
#   - the ArUco markers tell the software WHICH corner is which, so the board
#     still works when it is partly off the edge of the frame or partly hidden.
#
# "7 x 10" here means 7 squares ACROSS and 10 squares DOWN.
# That board has 7 x 10 = 70 squares, so it needs 70 / 2 = 35 ArUco markers.
# DICT_7X7_50 contains 50 markers, so it fits.

CHARUCO_SQUARES_X = 7          # squares across (the short side)
CHARUCO_SQUARES_Y = 10         # squares down  (the long side)

# The ArUco dictionary printed inside the chessboard squares.
# Kept the same as the rest of the project so there is only one to remember.
CHARUCO_DICTIONARY = cv2.aruco.DICT_7X7_50

# ---------------------------------------------------------------------
# THESE TWO NUMBERS ARE THE MOST IMPORTANT IN THIS FILE.
#
# They are the size of ONE chessboard square and ONE ArUco marker, in
# millimetres, ON THE PAPER YOU ACTUALLY PRINTED.
#
# Printers lie. "Fit to page", "shrink to printable area", and driver defaults
# all silently scale the page by a few percent. If you type the value you
# INTENDED to print instead of the value you MEASURED, every distance the pose
# tool reports will be wrong by that same percentage, and nothing will warn you.
#
# So: print the board, put a steel rule across FIVE squares, divide by five,
# and type that number here. Measuring five and dividing beats measuring one.
# ---------------------------------------------------------------------
CHARUCO_SQUARE_LENGTH_MM = 25.0     # <-- MEASURE THIS on the printed sheet
CHARUCO_MARKER_LENGTH_MM = 18.0     # <-- MEASURE THIS on the printed sheet

# Settings used only when GENERATING the printable board image.
PRINT_DPI = 600                      # dots per inch for the generated PNG
PRINT_MARGIN_MM = 10.0               # white border around the board


# ======================================================================
# SECTION 3 - Calibration capture and output
# ======================================================================

# Where captured calibration photographs are stored.
CALIBRATION_IMAGES_FOLDER = "calibration_images"

# Where the finished calibration is written, and read back from.
INTRINSICS_PATH = "camera_intrinsics.json"

# A photograph is only used if the detector finds at least this many chessboard
# corners in it. Views with very few corners add noise rather than information.
MIN_CHARUCO_CORNERS = 12

# Aim for at least this many usable photographs before calibrating. Fewer than
# about 15 gives an unstable answer; more than about 40 stops helping.
TARGET_CALIBRATION_IMAGES = 25

# After calibration, any single photograph whose own reprojection error exceeds
# this many pixels is reported as suspect (usually motion blur or a bent sheet).
SUSPECT_VIEW_ERROR_PIXELS = 1.0

# Use the "rational" distortion model (8 coefficients instead of 5)?
#
# The Facecam 4K is a wide lens (about 90 degrees), and wide lenses sometimes
# need the extra terms to fit properly. Leave this False first. If the overall
# reprojection error will not come below about 0.5 pixels, set it True and
# calibrate again - but only if you have plenty of photographs, because the
# extra coefficients need more data to pin down.
USE_RATIONAL_DISTORTION_MODEL = False


# ======================================================================
# SECTION 4 - The four ArUco markers on the plywood work surface
# ======================================================================
#
# These are the markers the delivered system already uses for rectification:
# IDs 0, 3, 4, 5 from DICT_7X7_50.
#
# To turn them into a POSE reference we need to know where they physically sit
# on the plywood. Measure once, carefully, with a tape measure or a long rule:
#
#   1. Pick one marker to be the ORIGIN. Its centre becomes (0, 0).
#      Use the one nearest the operator's left hand - it makes the numbers
#      easy to picture.
#   2. X increases to the operator's RIGHT.
#      Y increases AWAY from the operator.
#   3. For each of the other three markers, measure the X and Y distance from
#      the origin marker's centre to that marker's centre, in millimetres.
#   4. Measure the printed side length of one marker (the black square only,
#      not the white quiet zone around it).
#
# "rotation_deg" is how far that marker is turned ANTICLOCKWISE on the board,
# compared with being printed square to the X and Y axes. It is almost always
# 0, 90, 180 or 270. If you are not sure, leave them all at 0 and run
#     python measure_camera_pose.py --solve-rotations
# which tries all the combinations and tells you which one fits.
#
# The values below are PLACEHOLDERS. Replace every one of them.

MARKER_DICTIONARY = cv2.aruco.DICT_7X7_50

MARKER_LAYOUT = {
    0: {"center_mm": (0.0,   0.0),   "size_mm": 40.0, "rotation_deg": 0},
    3: {"center_mm": (700.0, 0.0),   "size_mm": 40.0, "rotation_deg": 0},
    4: {"center_mm": (0.0,   500.0), "size_mm": 40.0, "rotation_deg": 0},
    5: {"center_mm": (700.0, 500.0), "size_mm": 40.0, "rotation_deg": 0},
}

# ArUco detector tuning. Copied from the delivered config.py so marker
# detection behaves identically in both places.
ARUCO_THRESH_WIN_MIN = 3
ARUCO_THRESH_WIN_MAX = 53
ARUCO_THRESH_WIN_STEP = 4


# ======================================================================
# SECTION 5 - Pose reporting
# ======================================================================

# The point on the board that tilt and azimuth are reported ABOUT.
#
# "Tilt" only means something relative to a specific spot: the camera sits at
# one angle above the centre of the board and a different angle above its
# corner. Set this to None to use the centre of the four markers, which is
# almost always what you want.
POSE_REFERENCE_POINT_MM = None       # None = centroid of the four markers

# How far above the board surface the tracked colour on the probe sits, in
# millimetres. This is the 'h' from Section 5 of the Camera Position
# Evaluation Plan. Measure it: hold the probe as you would in normal use with
# its tip on a terminal, and measure from the plywood up to the vertical
# middle of the coloured region the tracker latches onto.
PROBE_COLOUR_HEIGHT_MM = 12.0        # <-- MEASURE THIS

# Board points at which the parallax report is printed. Set these to the real
# corners and centre of your working area so the report tells you the error at
# places you actually probe.
PARALLAX_SAMPLE_POINTS_MM = [
    ("centre",       (350.0, 250.0)),
    ("left edge",    (50.0,  250.0)),
    ("right edge",   (650.0, 250.0)),
    ("near corner",  (50.0,  50.0)),
    ("far corner",   (650.0, 450.0)),
]

# Where a measured pose is written when you press 's'.
POSE_OUTPUT_FOLDER = "camera_poses"
