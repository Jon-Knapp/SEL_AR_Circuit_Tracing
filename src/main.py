# main.py
#
# The live application for the Continuity Annotation System.
#
# Three vision stages, composed:
#   1. PROBE TRACKING (every frame, on the RAW feed). Tracks the two probes by
#      color. This is the surface you interact with and calibrate on.
#   2. COMPONENT DETECTION (once, on the RAW feed). Runs the YOLO model a single
#      time to find the panel parts, then draws those boxes every frame. The
#      panel does not move, so there is no reason to re-run YOLO live.
#   3. RECTIFICATION (a display/record surface). Finds four ArUco markers once,
#      flattens the board, and shows a second window. The component boxes,
#      probe tips, and terminal circles computed on the RAW feed are pushed
#      through the homography so they appear on the flattened board.
#
# THE TERMINAL MAP
#   At start-up this program loads the terminal-map TEMPLATE built by
#   calibrate_terminals.py. The template stores, per device class, where each
#   terminal sits inside that class's oriented bounding box. Every frame we
#   stamp those terminals onto EACH detected device, so a terminal circle
#   appears on every screw - including on extra devices of the same class the
#   template was never directly calibrated on. When a probe sits on a terminal,
#   that terminal is highlighted and its name is shown in the top-left status
#   (NOT printed over the board, to keep the view clean).
#
# CONTINUITY RECORDING (added)
#   The two probes are wired to a LabJack U12, which reports a single yes/no:
#   are the probe tips electrically connected? When the LabJack says "yes" AND
#   the camera sees each probe on a known, DIFFERENT terminal, and that holds
#   steady for a short debounce, we automatically RECORD that pair as a
#   connection. Connections are grouped (everything wired together shares one
#   colored flag + label like "G1"), written to a .txt table and a SQLite
#   database, and shown on the board. Press 'u' to undo the most recent
#   connection. See connection_log.py for the grouping logic.
#
#   Honest caveat (record integrity): a recorded pair is the pair the VISION
#   system believed the probes were on. The debounce rejects a momentary brush,
#   but it cannot catch a probe resting steadily on a MIS-identified terminal -
#   the record is only as trustworthy as the terminal IDs underneath it.
#
# Calibration lives in the separate calibrate_terminals.py tool; this program
# only USES the template.
#
# Controls are listed in the on-screen "Controls" window (toggle with 'h').

import os
import time
import cv2
import numpy as np
from datetime import datetime

import config
import probe_tracking as pt
import terminal_map as tm
import ui
import labjack_interface as lj
import connection_log as cl

# Component detection is optional. If ultralytics/object_detection can't be
# imported (or you set ENABLE_DETECTION = False), the program still runs with
# just probe tracking and rectification - but with no detections there are no
# device boxes, so no terminal circles can be stamped.
try:
    import object_detection as od
    from ultralytics import YOLO
    DETECTION_AVAILABLE = True
    DETECTION_IMPORT_ERROR = None
except Exception as error:           # pragma: no cover - depends on the machine
    od = None
    YOLO = None
    DETECTION_AVAILABLE = False
    DETECTION_IMPORT_ERROR = str(error)


CONTROLS_TITLE = "Main controls"
CONTROLS_SECTIONS = [
    ("Probes", [
        ("1 / 2", "choose which probe to teach"),
        ("click + s", "teach the chosen probe its color"),
        ("d", "toggle probe color-mask debug windows"),
    ]),
    ("View", [
        ("g", "show / hide the terminal circles + group flags"),
        ("h", "show / hide this controls window"),
    ]),
    ("Records", [
        ("c", "save clean record + labeled legend"),
        ("r", "start / stop recording"),
        ("u", "undo the last recorded connection"),
    ]),
    ("Setup", [
        ("o", "re-run component detection"),
        ("l", "re-lock the homography"),
        ("q", "save final record + quit"),
    ]),
]


# ======================================================================
# Rectification helpers
# ======================================================================

def order_corner_points(points):
    """Sort four (x, y) points into (top_left, top_right, bottom_right,
    bottom_left)."""
    points = np.array(points, dtype="float32")
    s = points[:, 0] + points[:, 1]
    d = points[:, 1] - points[:, 0]
    return np.array([
        points[np.argmin(s)], points[np.argmin(d)],
        points[np.argmax(s)], points[np.argmax(d)],
    ], dtype="float32")


def compute_homography(marker_centers):
    """Return (homography_matrix, (out_width, out_height)) that flattens the
    rectangle formed by the four marker centers."""
    src = order_corner_points(marker_centers)
    top_left, top_right, bottom_right, bottom_left = src
    width = int(max(np.linalg.norm(top_right - top_left),
                    np.linalg.norm(bottom_right - bottom_left)))
    height = int(max(np.linalg.norm(bottom_left - top_left),
                     np.linalg.norm(bottom_right - top_right)))
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix, _ = cv2.findHomography(src, dst)
    return matrix, (width, height)


def make_placeholder_image(width, height, message):
    """A dark image with a centered message, shown before the homography lock."""
    image = np.zeros((height, width, 3), dtype="uint8")
    cv2.putText(image, message, (30, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return image


def transform_points(points, homography):
    """Push raw-frame (x, y) point(s) through the homography into rectified
    coordinates. Accepts one point or a list of points; returns an (N, 2)
    array."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


# ======================================================================
# Detection helpers
# ======================================================================

def detect_components_once(model, operator_frame):
    """Run YOLO a single time and return component records in operator-frame
    (raw) pixel coordinates, with overlapping duplicates removed."""
    height, width = operator_frame.shape[:2]
    if config.CAMERA_FLIP_CODE is None:
        model_frame = operator_frame
    else:
        model_frame = cv2.flip(operator_frame, config.CAMERA_FLIP_CODE)

    results = model.predict(model_frame, conf=config.CONFIDENCE_THRESHOLD,
                            verbose=False)
    records = od.extract_detections(results[0], model.names,
                                    config.CAMERA_FLIP_CODE, width, height)
    return od.remove_overlapping_duplicates(records, config.OVERLAP_THRESHOLD)


def draw_components_rectified(rect_frame, components, homography, show_labels):
    """Draw the (raw-space) component boxes onto the flattened board."""
    for component in components:
        mapped = transform_points(component["corners"], homography)
        corners = [(int(x), int(y)) for x, y in mapped]
        color = od.BOX_COLORS[component["class_index"] % len(od.BOX_COLORS)]
        label = (f"{component['class_name']} {component['confidence']:.2f}"
                 if show_labels else None)
        od.draw_one_detection(rect_frame, corners, label, color)


def draw_probe_tips_rectified(rect_frame, records, homography):
    """Draw each found probe's tip onto the flattened board."""
    for record in records:
        if not record["found"]:
            continue
        mapped = transform_points([record["tip"]], homography)[0]
        tip = (int(mapped[0]), int(mapped[1]))
        color = record["draw_color"]
        cv2.circle(rect_frame, tip, 12, color, 2)
        cv2.circle(rect_frame, tip, 4, color, -1)
        cv2.putText(rect_frame, record["label"], (tip[0] + 15, tip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ======================================================================
# Terminal-map helpers
# ======================================================================

def draw_terminals_clean(frame, terminals, hit_ids):
    """Draw every terminal as a clean colored circle (no text). The terminals a
    probe is currently on are highlighted."""
    for terminal in terminals:
        color = od.BOX_COLORS[terminal["class_index"] % len(od.BOX_COLORS)]
        highlight = terminal["terminal_id"] in hit_ids
        tm.draw_terminal_circle(frame, (terminal["x"], terminal["y"]),
                                color, highlight=highlight)


def draw_terminals_clean_rectified(rect_frame, terminals, hit_ids, homography):
    """Same as draw_terminals_clean but mapped onto the flattened board."""
    if not terminals:
        return
    raw_points = [(t["x"], t["y"]) for t in terminals]
    mapped = transform_points(raw_points, homography)
    for terminal, (mx, my) in zip(terminals, mapped):
        color = od.BOX_COLORS[terminal["class_index"] % len(od.BOX_COLORS)]
        highlight = terminal["terminal_id"] in hit_ids
        tm.draw_terminal_circle(rect_frame, (mx, my), color, highlight=highlight)


# ======================================================================
# Connection-group flag helpers
# ======================================================================
#
# A "group flag" is how we DOCUMENT a connection on the board: every terminal
# that belongs to a connection group gets a bold ring in that group's color and
# the group's label (e.g. "G1") next to it. Terminals not in any group keep just
# their plain circle. When two groups merge, all their terminals end up showing
# the same label/color, which is exactly the "all connected together" view.
#
# This deliberately prints short labels ON the board (G1, G2, ...). That is a
# change from the otherwise label-free board, but the project asked for a
# unique, visible flag per connection, and the label is what makes each group
# unambiguous.

def draw_group_flags(frame, terminals, group_for_terminal):
    """Flag every terminal that belongs to a connection group, in raw pixels."""
    for terminal in terminals:
        group = group_for_terminal.get(terminal["terminal_id"])
        if group is None:
            continue
        x, y = int(terminal["x"]), int(terminal["y"])
        cv2.circle(frame, (x, y), 13, group["color"], 3)
        draw_text_with_outline(frame, group["label"], (x + 14, y - 10),
                               group["color"], scale=0.6)


def draw_group_flags_rectified(rect_frame, terminals, group_for_terminal,
                               homography):
    """Same as draw_group_flags but mapped onto the flattened board."""
    flagged = [t for t in terminals
               if t["terminal_id"] in group_for_terminal]
    if not flagged:
        return
    raw_points = [(t["x"], t["y"]) for t in flagged]
    mapped = transform_points(raw_points, homography)
    for terminal, (mx, my) in zip(flagged, mapped):
        group = group_for_terminal[terminal["terminal_id"]]
        x, y = int(mx), int(my)
        cv2.circle(rect_frame, (x, y), 13, group["color"], 3)
        draw_text_with_outline(rect_frame, group["label"], (x + 14, y - 10),
                               group["color"], scale=0.6)


def build_legend_image(clean_raw_frame, terminals, group_for_terminal):
    """Make the LABELED reference image: a clean raw frame with every terminal's
    ID printed next to it, plus the connection-group flags. Kept beside the JSON
    / database records so a person can match a terminal name to its place on the
    board and see which group it ended up in."""
    legend = clean_raw_frame.copy()
    for terminal in terminals:
        color = od.BOX_COLORS[terminal["class_index"] % len(od.BOX_COLORS)]
        tm.draw_terminal_label(legend, (terminal["x"], terminal["y"]),
                               terminal["terminal_id"], color)
    draw_group_flags(legend, terminals, group_for_terminal)
    return legend


# ======================================================================
# Overlay helpers
# ======================================================================

def make_timestamped_filename(folder, prefix, extension):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(folder, f"{prefix}_{timestamp}.{extension}")


def draw_text_with_outline(frame, text, origin, color, scale=0.6):
    """Bright text with a black outline, so it reads over the busy background."""
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def draw_labjack_warning(frame):
    """A bold red 'LABJACK NOT CONNECTED' banner across the top. Drawn whenever
    the continuity sensor is unavailable, INCLUDING on saved records, so a saved
    image never silently implies continuity was being measured when it was not."""
    text = "LABJACK NOT CONNECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 3
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (frame.shape[1] - text_w) // 2)
    y = 60
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 4,
                cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 255), thickness,
                cv2.LINE_AA)


def draw_sampling_overlay(frame, active_label, last_sample_bgr):
    """Top-left: which probe a click will teach, plus the last clicked color."""
    text = f"Click teaches: {active_label}   (1/2 choose, click, 's' save)"
    draw_text_with_outline(frame, text, (20, 32), (255, 255, 255))
    if last_sample_bgr is not None:
        cv2.rectangle(frame, (20, 42), (70, 92), last_sample_bgr, -1)
        cv2.rectangle(frame, (20, 42), (70, 92), (255, 255, 255), 1)
        cv2.putText(frame, f"last click BGR={last_sample_bgr}", (80, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_status_table(frame, records, homography_status, detected,
                      terminal_count, labjack_status, connection_count,
                      group_count):
    """Top-right: probe status, homography / detection state, terminal count,
    LabJack status, and connection / group counts."""
    rows = [f"{r['label']}: {'DETECTED' if r['found'] else 'not found'}"
            for r in records]
    if config.ENABLE_RECTIFICATION:
        rows.append(f"Homography: {homography_status}")
    if config.ENABLE_DETECTION:
        rows.append(f"Components: {'found' if detected else 'pending'}")
    rows.append(f"Terminals: {terminal_count}")
    rows.append(f"LabJack: {labjack_status}")
    rows.append(f"Connections: {connection_count}")
    rows.append(f"Groups: {group_count}")
    if not rows:
        return

    font, scale, thick, line_h, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 20, 8
    box_w = max(cv2.getTextSize(r, font, scale, thick)[0][0] for r in rows) + 2 * pad
    box_h = line_h * len(rows) + 2 * pad
    fw = frame.shape[1]
    x1, y1 = fw - box_w - 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, row in enumerate(rows):
        cv2.putText(frame, row, (x1 + pad, y1 + pad + line_h * (i + 1) - 4),
                    font, scale, (255, 255, 255), thick)


# ======================================================================
# Saving records (used by both 'c' and 'q')
# ======================================================================

def save_records(rect_annotated, working, operator_frame, terminals,
                 group_for_terminal, labjack_connected):
    """Save two images: the CLEAN record (rectified if available, else the raw
    working view) and the LABELED legend. Both already carry the group flags;
    the legend additionally gets terminal-ID labels. If the LabJack is offline,
    the warning banner is stamped on the legend too. Returns the two paths."""
    if rect_annotated is not None:
        clean_path = make_timestamped_filename(config.CAPTURES_FOLDER,
                                                "record_rectified", "jpg")
        cv2.imwrite(clean_path, rect_annotated)
    else:
        clean_path = make_timestamped_filename(config.CAPTURES_FOLDER,
                                                "record", "jpg")
        cv2.imwrite(clean_path, working)

    legend = build_legend_image(operator_frame, terminals, group_for_terminal)
    if not labjack_connected:
        draw_labjack_warning(legend)
    legend_path = make_timestamped_filename(config.CAPTURES_FOLDER,
                                            "legend", "jpg")
    cv2.imwrite(legend_path, legend)
    return clean_path, legend_path


# ======================================================================
# Main
# ======================================================================

def main():
    os.makedirs(config.CAPTURES_FOLDER, exist_ok=True)
    os.makedirs(config.RECORDINGS_FOLDER, exist_ok=True)

    # --- Decide which optional stages are actually available ---
    detection_on = config.ENABLE_DETECTION
    if detection_on and not DETECTION_AVAILABLE:
        print("Component detection requested but could not be loaded:")
        print(f"   {DETECTION_IMPORT_ERROR}")
        print("Continuing with probe tracking only (no terminal circles).")
        detection_on = False

    # --- Open the camera ---
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open camera at index {config.CAMERA_INDEX}.")
        return

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    print(f"Camera: {actual_width} x {actual_height} @ {fps:.0f} fps")

    # --- Load the YOLO model once (only if detection is on) ---
    model = None
    if detection_on:
        try:
            print("Loading component-detection model...")
            model = YOLO(config.MODEL_PATH)
            print("Model loaded.")
        except Exception as error:
            print(f"Could not load model '{config.MODEL_PATH}': {error}")
            print("Continuing with probe tracking only.")
            detection_on = False

    # --- Build the ArUco detector (only if we will rectify) ---
    aruco_detector = None
    if config.ENABLE_RECTIFICATION:
        aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICTIONARY)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = config.ARUCO_THRESH_WIN_MIN
        params.adaptiveThreshWinSizeMax = config.ARUCO_THRESH_WIN_MAX
        params.adaptiveThreshWinSizeStep = config.ARUCO_THRESH_WIN_STEP
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # --- Prepare the probes ---
    for probe in config.PROBES:
        probe["prev_center"] = None
        pt.recompute_ranges(probe)

    # --- Load the terminal-map template ---
    template_devices = tm.load_template(config.TERMINAL_MAP_PATH)
    if template_devices:
        print(f"Loaded terminal template for classes: "
              f"{sorted(template_devices.keys())}")
    else:
        print(f"No terminal template found at {config.TERMINAL_MAP_PATH}. "
              f"Run calibrate_terminals.py first to create one.")

    # --- Set up continuity sensing (LabJack U12) ---
    labjack = lj.LabJackContinuity(config.LABJACK_CHANNEL,
                                   config.LABJACK_CONTINUITY_STATE)
    if labjack.connected:
        print("LabJack U12 connected; continuity sensing is active.")
    else:
        print("WARNING: LabJack U12 not detected; continuity will not be "
              "recorded.")
        print(f"   Reason: {labjack.last_error}")
        print("   The program keeps running; a red on-screen warning will show.")

    # --- Set up the connection log (fresh, timestamped files per session) ---
    os.makedirs(config.CONNECTIONS_FOLDER, exist_ok=True)
    session_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_path = os.path.join(config.CONNECTIONS_FOLDER,
                            f"connections_{session_stamp}.txt")
    db_path = os.path.join(config.CONNECTIONS_FOLDER,
                           f"connections_{session_stamp}.db")
    log = cl.ConnectionLog(txt_path, db_path, config.GROUP_COLORS)
    print("Connection records this session:")
    print(f"   text:   {txt_path}")
    print(f"   sqlite: {db_path}")

    # --- Mouse callback state ---
    state = {"active_idx": 0, "last_sample": None, "surface": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["surface"] is None:
            return
        sample = pt.sample_bgr_from_click(state["surface"], x, y)
        if sample is None:
            return
        state["last_sample"] = sample
        hsv = cv2.cvtColor(np.uint8([[sample]]), cv2.COLOR_BGR2HSV)[0, 0]
        print(f"[click] ({x},{y})  BGR={sample}  HSV={tuple(int(c) for c in hsv)}")

    WORKING_WINDOW = "Working view (raw)"
    RECORD_WINDOW = "Rectified record"
    cv2.namedWindow(WORKING_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WORKING_WINDOW, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    cv2.setMouseCallback(WORKING_WINDOW, on_mouse)
    if config.ENABLE_RECTIFICATION:
        cv2.namedWindow(RECORD_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RECORD_WINDOW, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

    controls_open = True
    ui.show_controls_window("Controls", CONTROLS_TITLE, CONTROLS_SECTIONS)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- State carried between frames ---
    homography_locked = False
    homography_matrix = None
    homography_size = None

    accumulated_centers = {}
    seen_unexpected = set()
    last_progress_key = None

    components = []
    components_detected = False
    last_detection_counts = None    # last counts we printed while searching
    terminals = []                  # stamped terminal records, rebuilt on detect

    video_writer = None
    current_video_path = None
    record_size = (int(actual_width * config.RECORD_SCALE),
                   int(actual_height * config.RECORD_SCALE))

    show_debug = False
    debug_windows_open = False
    show_terminals = True           # 'g' toggles the terminal circles + flags

    # --- Continuity / debounce state ---
    labjack_continuity = False       # the last continuity reading (cached)
    last_labjack_poll = 0.0          # when we last read the LabJack
    last_reconnect_attempt = 0.0     # when we last tried to reopen it
    candidate_pair = None            # the pair currently being watched
    candidate_since = 0.0            # when the current candidate first appeared
    committed_this_touch = False     # have we already recorded this touch?

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to grab a frame. Exiting.")
            break

        operator_frame = frame      # the Elgato already delivers operator view
        now = time.time()

        # --- Stage 2: run YOLO every frame until the FULL expected set of
        # devices appears in a SINGLE frame, then lock it (mirrors how the
        # ArUco markers accumulate before the homography locks). The board is
        # static, so once one clean frame shows every expected device, that
        # frame is trustworthy and we stop re-running YOLO. ---
        if detection_on and not components_detected:
            components = detect_components_once(model, operator_frame)
            counts = od.count_detections_by_class(components)

            if counts != last_detection_counts:
                print(f"Detection: have {dict(sorted(counts.items()))}  "
                      f"need {dict(sorted(config.EXPECTED_DEVICE_COUNTS.items()))}")
                last_detection_counts = counts

            if counts == config.EXPECTED_DEVICE_COUNTS:
                components_detected = True
                print("Components locked (full expected set detected).")
                terminals = tm.apply_template(components, template_devices)
                print(f"Stamped {len(terminals)} terminals from the template.")

        # --- Stage 3: accumulate ArUco markers across frames, then lock ---
        if config.ENABLE_RECTIFICATION and not homography_locked:
            gray = cv2.cvtColor(operator_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)
            if ids is not None:
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    marker_id = int(marker_id)
                    center = marker_corners.reshape(4, 2).mean(axis=0)
                    if marker_id in config.EXPECTED_MARKER_IDS:
                        accumulated_centers[marker_id] = center
                    else:
                        seen_unexpected.add(marker_id)

            missing = sorted(config.EXPECTED_MARKER_IDS - set(accumulated_centers))
            progress_key = (tuple(sorted(accumulated_centers)),
                            tuple(sorted(seen_unexpected)))
            if progress_key != last_progress_key:
                message = (f"ArUco: have {sorted(accumulated_centers)}  "
                           f"missing {missing}")
                if seen_unexpected:
                    message += (f"  | also seeing unexpected ids "
                                f"{sorted(seen_unexpected)} - add to "
                                f"EXPECTED_MARKER_IDS if one is a real corner")
                print(message)
                last_progress_key = progress_key

            if not missing:
                homography_matrix, homography_size = compute_homography(
                    list(accumulated_centers.values()))
                homography_locked = True
                print("Homography locked.")

        # --- Stage 1: track the probes on the RAW feed (always) ---
        state["surface"] = operator_frame.copy()
        hsv = cv2.cvtColor(operator_frame, cv2.COLOR_BGR2HSV)
        records = [pt.track_probe(hsv, probe, operator_frame.shape, kernel,
                                  config.PROBE_MAX_JUMP_FRACTION)
                   for probe in config.PROBES]

        # --- Which terminal is each probe on? ---
        terminal_hits = []
        for record in records:
            if record["found"]:
                hit = tm.nearest_terminal(record["tip"], terminals,
                                          config.TERMINAL_MATCH_MAX_DISTANCE)
            else:
                hit = None
            terminal_hits.append(hit)
        hit_ids = {h["terminal_id"] for h in terminal_hits if h is not None}

        # --- Continuity sensing + automatic, debounced recording ---
        # Read the LabJack on an interval (its USB read is too slow to do every
        # frame). Reuse the cached value in between reads.
        if now - last_labjack_poll >= config.LABJACK_POLL_INTERVAL:
            last_labjack_poll = now
            reading = labjack.read_continuity()       # True / False / None
            labjack_continuity = (reading is True)
            # If it looks gone, try (gently, infrequently) to reopen it.
            if (not labjack.connected and
                    now - last_reconnect_attempt >= config.LABJACK_RECONNECT_INTERVAL):
                last_reconnect_attempt = now
                labjack.try_open()

        # A connection is only real when BOTH probes are on KNOWN, DIFFERENT
        # terminals AND the LabJack reports continuity. Otherwise there is no
        # pair we could honestly record.
        red_hit = terminal_hits[0] if len(terminal_hits) > 0 else None
        black_hit = terminal_hits[1] if len(terminal_hits) > 1 else None
        current_pair = None
        if (labjack.connected and labjack_continuity
                and red_hit is not None and black_hit is not None
                and red_hit["terminal_id"] != black_hit["terminal_id"]):
            a, b = sorted([red_hit["terminal_id"], black_hit["terminal_id"]])
            current_pair = (a, b)

        # Debounce: the SAME pair must hold steady for a short time before we
        # trust it. This rejects a momentary brush. When the situation changes
        # (different pair, or probes lifted), the timer restarts.
        if current_pair != candidate_pair:
            candidate_pair = current_pair
            candidate_since = now
            committed_this_touch = False

        if (candidate_pair is not None and not committed_this_touch
                and now - candidate_since >= config.CONNECTION_DEBOUNCE_SECONDS):
            was_new = log.add_connection(candidate_pair[0], candidate_pair[1])
            committed_this_touch = True       # don't re-record until this touch ends
            if was_new:
                print(f"Connection recorded: {candidate_pair[0]} <-> "
                      f"{candidate_pair[1]}")
            else:
                print(f"Connection already on record: {candidate_pair[0]} <-> "
                      f"{candidate_pair[1]}")

        # --- Build the WORKING view (raw + overlays) ---
        working = operator_frame.copy()
        if detection_on and components:
            od.draw_detections(working, components, config.SHOW_LABELS)
        for record in records:
            pt.draw_probe(working, record)
        if show_terminals:
            draw_terminals_clean(working, terminals, hit_ids)
            draw_group_flags(working, terminals, log.group_for_terminal)

        active_label = config.PROBES[state["active_idx"]]["label"]
        draw_sampling_overlay(working, active_label, state["last_sample"])

        # Show the terminal(s) a probe is on as TEXT in the corner only, so the
        # board itself stays clean.
        active_names = [h["terminal_id"] for h in terminal_hits if h is not None]
        on_text = "On: " + (", ".join(active_names) if active_names else "-")
        draw_text_with_outline(working, on_text, (20, 120), (0, 255, 255), scale=0.7)

        if not config.ENABLE_RECTIFICATION:
            homography_status = "off"
        elif homography_locked:
            homography_status = "locked"
        else:
            have = len(accumulated_centers)
            missing = sorted(config.EXPECTED_MARKER_IDS - set(accumulated_centers))
            homography_status = f"{have}/4 (missing {missing})"

        labjack_status = "connected" if labjack.connected else "NOT CONNECTED"
        draw_status_table(working, records, homography_status,
                          components_detected, len(terminals), labjack_status,
                          len(log.connections), len(log.groups))

        # Persistent, hard-to-miss warning if the continuity sensor is offline.
        if not labjack.connected:
            draw_labjack_warning(working)

        # --- Build the RECTIFIED record view ---
        rect_annotated = None
        if config.ENABLE_RECTIFICATION:
            if homography_locked:
                rect_annotated = cv2.warpPerspective(
                    operator_frame, homography_matrix, homography_size)
                if detection_on and components:
                    draw_components_rectified(rect_annotated, components,
                                              homography_matrix, config.SHOW_LABELS)
                draw_probe_tips_rectified(rect_annotated, records, homography_matrix)
                if show_terminals:
                    draw_terminals_clean_rectified(rect_annotated, terminals,
                                                   hit_ids, homography_matrix)
                    draw_group_flags_rectified(rect_annotated, terminals,
                                               log.group_for_terminal,
                                               homography_matrix)
                if not labjack.connected:
                    draw_labjack_warning(rect_annotated)
                cv2.imshow(RECORD_WINDOW, rect_annotated)
            else:
                cv2.imshow(RECORD_WINDOW, make_placeholder_image(
                    640, 480, f"Finding ArUco markers... {homography_status}"))

        # --- Record the working view ---
        if video_writer is not None:
            video_writer.write(cv2.resize(working, record_size))

        # --- On-screen preview (REC dot added here only) ---
        display = working.copy()
        if video_writer is not None:
            cv2.circle(display, (actual_width - 40, actual_height - 40),
                       12, (0, 0, 255), -1)
            cv2.putText(display, "REC", (actual_width - 120, actual_height - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(WORKING_WINDOW, display)

        # --- Debug mask windows ---
        if show_debug and records:
            for record in records:
                raw_bgr = cv2.cvtColor(record["raw_mask"], cv2.COLOR_GRAY2BGR)
                clean_bgr = cv2.cvtColor(record["clean_mask"], cv2.COLOR_GRAY2BGR)
                cv2.putText(raw_bgr, "RAW", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(clean_bgr, "CLEAN", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                pair = np.hstack([raw_bgr, clean_bgr])
                pair = cv2.resize(pair, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT // 2))
                cv2.imshow(f"Debug mask - {record['label']}", pair)
            debug_windows_open = True
        elif debug_windows_open:
            for probe in config.PROBES:
                cv2.destroyWindow(f"Debug mask - {probe['label']}")
            debug_windows_open = False

        # --- Keyboard ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Save the final annotated record before quitting.
            clean_path, legend_path = save_records(
                rect_annotated, working, operator_frame, terminals,
                log.group_for_terminal, labjack.connected)
            print(f"Final annotated record saved: {clean_path}")
            print(f"Final labeled legend saved:   {legend_path}")
            break

        elif key == ord('1'):
            state["active_idx"] = 0
            print(f"Click will teach: {config.PROBES[0]['label']}")

        elif key == ord('2') and len(config.PROBES) > 1:
            state["active_idx"] = 1
            print(f"Click will teach: {config.PROBES[1]['label']}")

        elif key == ord('s'):
            if state["last_sample"] is None:
                print("No color sampled yet. Click on a probe first.")
            else:
                probe = config.PROBES[state["active_idx"]]
                probe["seed_bgr"] = state["last_sample"]
                probe["prev_center"] = None
                pt.recompute_ranges(probe)
                print(f"Taught {probe['label']} the color BGR={state['last_sample']}")

        elif key == ord('g'):
            show_terminals = not show_terminals
            print(f"Terminal circles + group flags: "
                  f"{'ON' if show_terminals else 'off'}")

        elif key == ord('h'):
            controls_open = not controls_open
            if controls_open:
                ui.show_controls_window("Controls", CONTROLS_TITLE, CONTROLS_SECTIONS)
            else:
                cv2.destroyWindow("Controls")

        elif key == ord('c'):
            clean_path, legend_path = save_records(
                rect_annotated, working, operator_frame, terminals,
                log.group_for_terminal, labjack.connected)
            print(f"Saved record: {clean_path}")
            print(f"Saved legend: {legend_path}")

        elif key == ord('r'):
            if video_writer is None:
                path = make_timestamped_filename(config.RECORDINGS_FOLDER,
                                                 "session", "mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(path, fourcc, fps, record_size)
                current_video_path = path
                print(f"Recording started: {path}")
            else:
                video_writer.release()
                video_writer = None
                print(f"Recording saved: {current_video_path}")

        elif key == ord('u'):
            removed = log.undo_last()
            if removed is not None:
                print(f"Undid connection: {removed['terminal_a']} <-> "
                      f"{removed['terminal_b']}")
                # Block an instant re-record while the probes are still touching
                # the same pair: keep this pair "already committed" until the
                # probes move off it. Lift and re-touch to record it again.
                candidate_pair = current_pair
                committed_this_touch = True
            else:
                print("No connections to undo.")

        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug masks: {'ON' if show_debug else 'off'}")


        elif key == ord('o') and detection_on:
            components_detected = False  # re-enter the search for the full set
            last_detection_counts = None  # so the search progress prints again
            print("Re-running component detection (waiting for the full set)...")

        elif key == ord('l') and config.ENABLE_RECTIFICATION:
            homography_locked = False
            homography_matrix = None
            homography_size = None
            accumulated_centers = {}
            seen_unexpected = set()
            last_progress_key = None
            print("Re-locking: re-finding the ArUco markers from scratch...")

    # --- Cleanup ---
    if video_writer is not None:
        video_writer.release()
        print(f"Recording saved: {current_video_path}")
    camera.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()