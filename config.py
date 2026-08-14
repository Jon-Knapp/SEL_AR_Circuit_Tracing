# config.py
#
# Central home for every "knob" in the Continuity Annotation System.
#
# If you want to change a SETTING (a number, a path, a toggle), change it
# HERE. You should not need to edit main.py or the tracking modules just to
# adjust a value.
#
#   - main.py reads the camera, ArUco, display, record, probe, LabJack, and
#     connection-logging settings.
#   - object_detection.py holds the YOLO component-detection functions.
#   - probe_tracking.py holds the color-based probe-tracking functions.
#   - terminal_map.py / calibrate_terminals.py read the terminal-map settings.
#   - labjack_interface.py reads the continuity sensor (settings passed in).
#   - connection_log.py records connections (settings passed in).

import cv2

# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------

# Which camera to open (0, 1, 2, ...). The Elgato is usually 1 when the
# machine also has a built-in webcam.
CAMERA_INDEX = 1

# Capture resolution. 4K uses the full Elgato sensor. Drop to 1920 x 1080
# if your machine struggles with the live feed.
CAPTURE_WIDTH = 3840
CAPTURE_HEIGHT = 2160

# How the Elgato Camera Hub is transforming the image, expressed as an
# OpenCV flip code:
#     1  = Mirror only        (left-right)
#     0  = Flip only          (top-bottom)
#    -1  = Mirror AND Flip     (180-degree rotation)   <-- current setting
#  None  = no transform
#
# The YOLO model was trained on the ORIGINAL (un-transformed) orientation,
# so object_detection.py undoes this before running the model. Color-based
# probe tracking does NOT care about orientation (a color is the same color
# whichever way the image is turned), so main.py tracks on the frame exactly
# as the operator sees it.
CAMERA_FLIP_CODE = -1

# ----------------------------------------------------------------------
# What main.py does - feature switches
# ----------------------------------------------------------------------

# IMPORTANT: probe tracking ALWAYS runs on the raw camera feed, because color
# tracking behaves best on un-warped pixels. These two switches add the other
# two vision stages on TOP of that.

# When True, main.py finds the four ArUco markers once, computes a homography
# that flattens the panel, locks it, and opens a second "Rectified record"
# window. The component boxes and probe tips that were computed in the raw
# feed are pushed through the homography so they appear on the flattened board.
# Nothing is TRACKED on the warped image - it is only a display/record surface.
ENABLE_RECTIFICATION = True

# When True, main.py runs the YOLO component detector ONCE (the panel parts
# are static, so there is no reason to re-run it every frame) and draws the
# component boxes every frame. Press 'o' while running to re-run detection.
ENABLE_DETECTION = True

# ----------------------------------------------------------------------
# ArUco markers / homography
# ----------------------------------------------------------------------

# The four marker IDs we expect, one per corner of the work surface. Any
# other markers in view are ignored.
EXPECTED_MARKER_IDS = {0, 3, 4, 5}

# The ArUco dictionary the markers were generated from.
ARUCO_DICTIONARY = cv2.aruco.DICT_7X7_50

# ArUco detector adaptive-threshold window range. Widening this range makes
# marker detection more tolerant of glare and shadows.
ARUCO_THRESH_WIN_MIN = 3
ARUCO_THRESH_WIN_MAX = 53
ARUCO_THRESH_WIN_STEP = 4

# ----------------------------------------------------------------------
# Object detection (YOLO) - used by object_detection.py
# ----------------------------------------------------------------------

# Path to the trained YOLO oriented-bounding-box (OBB) model.
MODEL_PATH = "weights_v2_4_obb.pt"

# Minimum confidence (0.0 - 1.0) for a detection to be kept.
CONFIDENCE_THRESHOLD = 0.522 #All classes at 0.95 at this confidence threshold
                            # according to F1-Confidence Curve

# When two boxes of the SAME class overlap more than this (IoU, 0.0 - 1.0),
# the weaker one is treated as a duplicate and dropped.
OVERLAP_THRESHOLD = 0.5

# Show the device name + confidence text above each box?
SHOW_LABELS = False

# Line thickness for the drawn component boxes.
BOX_THICKNESS = 3

# Box colors in OpenCV's (Blue, Green, Red) order, one cycled per class.
BOX_COLORS = [
    (0, 255, 0),    # green
    (255, 128, 0),  # blue-ish
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 128, 255),  # orange
    (255, 255, 0),  # cyan
]

# ----------------------------------------------------------------------
# Probe tracking (color-based) - used by probe_tracking.py
# ----------------------------------------------------------------------

# A probe is not allowed to jump more than this fraction of the frame WIDTH
# between consecutive frames. This rejects a far-away object of the same
# color from stealing the track. Raise it if a fast hand movement makes a
# probe "drop"; lower it if a distractor still gets grabbed.
PROBE_MAX_JUMP_FRACTION = 0.25

# Each probe is described by one dictionary. You do NOT need to guess the
# colors: run main.py, press 1 or 2 to choose a probe, CLICK on it in the
# live video, then press 's' to bake that color in. seed_bgr is only a
# starting point so something is tracked before your first click.
#
# Field guide:
#   label       : name drawn on screen / stored with the record
#   seed_bgr    : starting color guess in (Blue, Green, Red) order
#   draw_color  : (B, G, R) color used to draw THIS probe's overlay
#   h_tol/s_tol/v_tol : how far around the sampled color we still accept,
#                       in Hue / Saturation / Value. Wider = more forgiving
#                       but more prone to noise.
#   s_min/v_min : floors that stop the range from sliding into grey/black,
#                 where color (Hue) is unreliable.
#   open_iter/close_iter : mask clean-up strength (see probe_tracking.py).
#   min_area/max_area : ignore blobs smaller/larger than this many pixels.
#   tip_method  : how to turn the blob into a single contact point:
#                   "centroid" -> center of the blob. Best for a SHORT piece
#                                 of tape sitting near the metal tip.
#                   "axis_end" -> far end of the blob's long axis. Better for
#                                 a LONG colored body, but see the caveat in
#                                 probe_tracking.py (assumes the hand enters
#                                 from a frame edge).
#
# NOTE: the terminal map is calibrated and matched on the probe CENTROID, so
# the red probe (the one you calibrate with) should keep tip_method "centroid"
# so the map and the live readout agree.

PROBES = [
    {
        "label": "Red probe",
        "seed_bgr": (181, 82, 19),       # reddish; click to refine
        "draw_color": (0, 0, 255),       # draw in red
        "h_tol": 10, "s_tol": 90, "v_tol": 90,
        "s_min": 80, "v_min": 60,
        "open_iter": 2, "close_iter": 2,
        "min_area": 150, "max_area": 80000,
        "tip_method": "centroid",
    },
    {
        "label": "Black probe",
        "seed_bgr": (146, 42, 148),      # magenta; click to refine
        "draw_color": (255, 0, 255),     # draw in magenta
        "h_tol": 12, "s_tol": 90, "v_tol": 90,
        "s_min": 80, "v_min": 60,
        "open_iter": 2, "close_iter": 2,
        "min_area": 100, "max_area": 80000,
        "tip_method": "centroid",
    },
]

# ----------------------------------------------------------------------
# Display + saved records
# ----------------------------------------------------------------------

# On-screen size for the preview window.
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Where captured record images are written.
CAPTURES_FOLDER = "captures"

# Where recorded videos are written.
RECORDINGS_FOLDER = "recordings"

# Recorded video is scaled DOWN from the capture resolution by this factor so
# real-time recording stays light. The aspect ratio is preserved.
#   1.0  = full capture resolution (heaviest)
#   0.5  = half resolution (e.g. 4K -> 1920 x 1080)
#   0.25 = quarter resolution (lightest)
RECORD_SCALE = 0.5

# ----------------------------------------------------------------------
# Terminal map (template-based) - used by terminal_map.py,
# calibrate_terminals.py, and main.py
# ----------------------------------------------------------------------
#
# The terminal map stores, for ONE device of each class, where its terminals
# sit INSIDE that device's oriented bounding box (as fractions u, v). At run
# time main.py re-detects every device and stamps those terminals onto each
# one, so the same calibration works for any device of that class placed
# anywhere inside the ArUco area. See terminal_map.py for the full explanation.

# Where the calibrated template is written / read.
TERMINAL_MAP_PATH = "terminal_map.json"

# Which key, in calibrate_terminals.py, selects which device class to mark.
CALIBRATION_DEVICE_KEYS = {
    "f": "Flathead_Block",
    "p": "Phillips_Block",
    "j": "Terminal_1",
    "k": "Terminal_2",
}

# When matching a live probe tip to a terminal, only accept the match if the
# tip is within this many pixels of a stamped terminal. Keep it a bit smaller
# than the spacing between neighbouring terminals. Tune after you see the real
# spacing in your captured images.
TERMINAL_MATCH_MAX_DISTANCE = 40

# ----------------------------------------------------------------------
# Device instance identity - used by terminal_map.InstanceRegistry
# ----------------------------------------------------------------------
#
# Without this, a device's "instance number" (the "4" in Terminal_2_4) was
# just its rank in a left-to-right sort, so a few pixels of detector jitter
# between re-detects could swap two devices' numbers, and a missed detection
# could shift every later number down by one - silently corrupting any
# connection already logged under those names. InstanceRegistry (in
# terminal_map.py) fixes this by remembering where each numbered device
# physically is and matching new detections to that memory instead of to
# sort order. These two constants tune how forgiving that matching is.

# When re-detecting, a device is considered "the same physical device we
# already numbered" if its center is within this many pixels of where that
# numbered device was last seen. Set it SMALLER than the spacing between two
# neighbouring devices of the same class, or two different blocks could claim
# each other's number. Measure that spacing on a captured 4K frame first.
DEVICE_MATCH_MAX_DISTANCE = 150

# Used ONLY when handing out numbers to devices seen for the first time, to
# put them in a sensible reading order. Devices whose centers are within this
# many pixels vertically are treated as being in the same ROW, and within a row
# they are numbered left-to-right. Set it to roughly half the vertical gap
# between your two rows of green terminal blocks.
DEVICE_ROW_TOLERANCE = 300

# ----------------------------------------------------------------------
# Continuity sensor (LabJack U12) - used by labjack_interface.py
# ----------------------------------------------------------------------

# Which U12 digital input the probes are read on (matches test_labjack_u12.py).
LABJACK_CHANNEL = 0

# The raw digital state that means "the probes are connected" (continuity).
# test_labjack_u12.py treats state == 1 as connected. If your wiring reads the
# opposite way (continuity shows up as 0), change this to 0.
LABJACK_CONTINUITY_STATE = 1

# How often to read the LabJack, in seconds. Its USB read is too slow to do on
# every video frame without dragging the live view, so we read on this interval
# and reuse the last value in between. 0.1 = ten reads per second.
LABJACK_POLL_INTERVAL = 0.1

# If the LabJack disconnects mid-session, how often (seconds) to quietly try to
# reopen it. Kept slow on purpose, so a missing device is not hammered.
LABJACK_RECONNECT_INTERVAL = 3.0

# ----------------------------------------------------------------------
# Connection logging - used by connection_log.py and main.py
# ----------------------------------------------------------------------

# A connection is recorded automatically once the SAME pair of terminals reads
# continuous for at least this many seconds. This short wait ("debounce")
# rejects a momentary accidental brush of the probes.
CONNECTION_DEBOUNCE_SECONDS = 0.10

# Where the per-session connection records (.txt and .db) are written. A fresh,
# timestamped pair of files is created each time main.py starts, so an old
# session's records are never overwritten.
CONNECTIONS_FOLDER = "connections"

# Colors (Blue, Green, Red order) used to flag connection GROUPS on screen.
# Group 1 uses the first color, group 2 the second, and so on. These are kept
# separate from the detector's BOX_COLORS so a group flag reads as its own
# layer; the on-screen "G1", "G2" labels make each group unambiguous regardless.
GROUP_COLORS = [
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
    (0, 200, 0),     # green
    (255, 0, 200),   # violet
    (0, 165, 255),   # orange
    (128, 128, 0),   # teal
    (180, 0, 255),   # pink-red
    (0, 128, 128),   # olive
]
