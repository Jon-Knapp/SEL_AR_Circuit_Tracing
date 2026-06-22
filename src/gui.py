# gui.py
#
# A PyQt5 GUI wrapper for the Continuity Annotation System.
#
# WHY THIS FILE EXISTS
#   main.py drives everything through keystrokes in OpenCV windows. This file
#   gives the SAME system a clickable interface for the Poster Day demo. Every
#   keystroke in main.py becomes a button here, and clicking on the video still
#   samples a probe's color exactly like before.
#
#   main.py is NOT modified and NOT run by this file. We only IMPORT it, to
#   borrow its drawing / geometry / save helpers so the GUI and the keystroke
#   version look pixel-for-pixel identical. If anything ever misbehaves in the
#   GUI, `python main.py` is still your untouched, working fallback.
#
# HOW A GUI WRAPS THE PIPELINE (the one big idea)
#   A GUI brings its own event loop (it sits waiting for clicks). OpenCV's
#   imshow/waitKey loop cannot share that thread, so we split the work in two:
#
#     * A BACKGROUND WORKER THREAD (CameraWorker) owns the camera and runs the
#       whole vision pipeline - the exact body of main.py's while-loop. It never
#       touches the GUI directly.
#     * The GUI THREAD just shows the finished frames and the status text, and
#       turns button clicks into simple "commands".
#
#   The two talk in only two safe ways:
#     1. The worker EMITS Qt "signals" (finished frame, rectified frame, status,
#        log messages). Qt delivers these to the GUI thread safely.
#     2. Buttons drop a COMMAND into a thread-safe mailbox (a queue.Queue). Once
#        per frame the worker checks its mailbox and acts - the same way it used
#        to check for a pressed key. This has no race conditions and is easy to
#        explain: "buttons leave notes; the worker reads its notes each frame."
#
#   Rule we never break: only the GUI thread touches GUI widgets; only the
#   worker thread touches the camera and OpenCV. That single rule is what keeps
#   a threaded GUI from crashing.
#
# WHAT MOVED FROM THE VIDEO INTO THE SIDE PANEL
#   In main.py the status table and the color-sampling overlay are drawn ON the
#   video. Here they live in the side panel instead, so the video pane stays
#   clean for the sponsor. Everything that belongs ON the board (component
#   boxes, probe markers, terminal circles, group flags, the "On:" text, and
#   the red LABJACK-NOT-CONNECTED banner) is still drawn on the frame, so the
#   SAVED records are unchanged - including the integrity-critical LabJack
#   banner that must never be missing from a saved image.

import queue
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------
# IMPORT ORDER MATTERS ON WINDOWS - read before reordering anything here.
#
# ultralytics is built on PyTorch, and PyTorch loads native Windows DLLs
# (c10.dll and friends) the moment it is imported. PyQt5 ALSO loads its own
# native DLLs. On Windows, whichever loads first wins: if PyQt5 loads first,
# PyTorch's c10.dll fails to initialize with "WinError 1114".
#
# So we import the detector (which pulls in PyTorch) UP HERE, BEFORE any PyQt5
# import below. Once torch's DLLs are loaded, PyQt5 can load safely. This is
# why the detector try/except sits above the Qt imports instead of with the
# other project modules. (main.py never imports PyQt5, so it never hit this.)
# ----------------------------------------------------------------------
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

# Qt imports come AFTER PyTorch has loaded its DLLs (see the note above).
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QRadioButton,
    QButtonGroup, QVBoxLayout, QHBoxLayout, QGroupBox, QDockWidget,
    QPlainTextEdit, QMessageBox, QSizePolicy,
)

# The project's own modules - the SAME ones main.py uses.
import config
import probe_tracking as pt
import terminal_map as tm
import labjack_interface as lj
import connection_log as cl

# We import main.py purely to reuse its helper functions (drawing, geometry,
# saving). Importing it does NOT run it - main()'s body only runs when main.py
# is launched directly (its `if __name__ == "__main__"` guard). Its own
# `from ultralytics import YOLO` is already satisfied by the import above, so
# this does not re-trigger the torch DLL load.
import main


# ======================================================================
# Small display helpers
# ======================================================================

def bgr_to_qpixmap(bgr_image):
    """
    Turn an OpenCV image (which stores colors in Blue-Green-Red order) into a
    QPixmap that a Qt label can show (Qt wants Red-Green-Blue order).

    The .copy() at the end is important: it makes Qt own its own copy of the
    pixels, so the original numpy array is free to be reused next frame without
    the displayed image turning to garbage.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    bytes_per_line = 3 * width
    image = QImage(rgb.data, width, height, bytes_per_line,
                   QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def fit_width(image, max_width):
    """Shrink an image so it is at most max_width pixels wide, keeping its shape.
    Used to keep the rectified preview light to send between threads."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    new_size = (max_width, int(round(height * scale)))
    return cv2.resize(image, new_size)


# ======================================================================
# The clickable video label
# ======================================================================

class ClickableLabel(QLabel):
    """A QLabel that reports where it was clicked. We use this for the main
    video pane so a click can sample a probe's color, just like clicking the
    OpenCV window in main.py."""

    clicked = pyqtSignal(int, int)          # (x, y) in label pixels

    def mousePressEvent(self, event):
        self.clicked.emit(event.x(), event.y())


# ======================================================================
# The background worker: owns the camera and runs the pipeline
# ======================================================================

class CameraWorker(QThread):
    """
    Runs the entire vision pipeline on its own thread. This is, almost line for
    line, the body of main.py's while-loop, with two differences:
      * instead of cv2.imshow it EMITS finished frames as signals, and
      * instead of reading a pressed key it reads COMMANDS from a mailbox.
    """

    # Signals the GUI listens to (Qt carries these safely across threads).
    main_frame_ready = pyqtSignal(object)     # the raw "working" view (numpy)
    rect_frame_ready = pyqtSignal(object)     # the rectified view (numpy)
    status_ready = pyqtSignal(dict)           # everything for the side panel
    message = pyqtSignal(str)                 # one line for the activity log
    failed = pyqtSignal(str)                  # a fatal start-up problem

    def __init__(self, main_view_width):
        super().__init__()
        # We downscale each finished frame to at most this width before sending
        # it to the GUI (keeps the cross-thread image light). The GUI then
        # scales that image to fit whatever size the video pane currently is,
        # so the video grows and shrinks with the window. Clicks are mapped back
        # using fractions, so this transport width does not affect click
        # accuracy - only how sharp the video looks when stretched large. Raise
        # it for a crisper picture on a big screen, at a little more CPU cost.
        self.main_view_width = main_view_width

        self._commands = queue.Queue()        # the mailbox the GUI writes to
        self._running = True                  # set False to ask the loop to stop

        # State the command handlers need. The loop fills these in each frame
        # before draining the mailbox, so a command always acts on fresh data.
        self._latest_operator_frame = None    # the raw frame, for color sampling
        self._active_idx = 0                  # which probe a click teaches
        self._last_sample = None              # last clicked BGR color
        self._recording = False               # are we writing a video right now?

    # ---- Called by the GUI thread to leave a command in the mailbox ----
    def send(self, command):
        """command is a tuple like ("undo",) or ("sample", u, v)."""
        self._commands.put(command)

    def request_stop(self):
        """Backup way to stop, used if a clean ("quit",) was not processed."""
        self._running = False

    # ------------------------------------------------------------------
    # The pipeline loop
    # ------------------------------------------------------------------
    def run(self):
        try:
            self._run_pipeline()
        except Exception as error:            # never let the thread die silently
            self.failed.emit(f"The camera worker stopped unexpectedly:\n{error}")

    def _run_pipeline(self):
        # --- Decide which optional stages are available ---
        detection_on = config.ENABLE_DETECTION
        if detection_on and not DETECTION_AVAILABLE:
            self.message.emit("Component detection could not be loaded; "
                              "running with probe tracking only.")
            self.message.emit(f"   reason: {DETECTION_IMPORT_ERROR}")
            detection_on = False

        # --- Open the camera (same settings as main.py) ---
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
        if not camera.isOpened():
            self.failed.emit(f"Could not open the camera at index "
                             f"{config.CAMERA_INDEX}.")
            return
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_fps = camera.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0 or camera_fps > 120:
            camera_fps = 30.0
        self.message.emit(f"Camera: {actual_width} x {actual_height} "
                          f"@ {camera_fps:.0f} fps")

        # --- Recording frame rate (Poster Day "Fix 2"). We PACE the writer by
        # the wall clock, so this number only affects smoothness/size, not
        # playback speed. If config has no RECORD_FPS yet, fall back safely. ---
        record_fps = getattr(config, "RECORD_FPS", None)
        if record_fps is None:
            record_fps = camera_fps
            self.message.emit("config.RECORD_FPS not set; using the camera fps "
                              "for recording.")

        # --- The full expected device set (Poster Day "Fix 1"). If config has
        # no EXPECTED_DEVICE_COUNTS yet, fall back to main.py's old behavior of
        # locking on the first detection. ---
        expected_counts = getattr(config, "EXPECTED_DEVICE_COUNTS", None)

        # --- Load the YOLO model once ---
        model = None
        if detection_on:
            try:
                self.message.emit("Loading component-detection model...")
                model = YOLO(config.MODEL_PATH)
                self.message.emit("Model loaded.")
            except Exception as error:
                self.message.emit(f"Could not load model "
                                  f"'{config.MODEL_PATH}': {error}")
                self.message.emit("Running with probe tracking only.")
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
            self.message.emit(f"Loaded terminal template for classes: "
                              f"{sorted(template_devices.keys())}")
        else:
            self.message.emit(f"No terminal template at "
                              f"{config.TERMINAL_MAP_PATH}. Run "
                              f"calibrate_terminals.py first.")

        # --- Continuity sensor (LabJack U12) ---
        labjack = lj.LabJackContinuity(config.LABJACK_CHANNEL,
                                       config.LABJACK_CONTINUITY_STATE)
        if labjack.connected:
            self.message.emit("LabJack U12 connected; continuity sensing active.")
        else:
            self.message.emit("WARNING: LabJack U12 not detected; continuity "
                              "will not be recorded.")
            self.message.emit(f"   reason: {labjack.last_error}")

        # --- Connection log (fresh, timestamped files per session) ---
        import os
        from datetime import datetime
        os.makedirs(config.CONNECTIONS_FOLDER, exist_ok=True)
        os.makedirs(config.CAPTURES_FOLDER, exist_ok=True)
        os.makedirs(config.RECORDINGS_FOLDER, exist_ok=True)
        session_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        txt_path = os.path.join(config.CONNECTIONS_FOLDER,
                                f"connections_{session_stamp}.txt")
        db_path = os.path.join(config.CONNECTIONS_FOLDER,
                               f"connections_{session_stamp}.db")
        log = cl.ConnectionLog(txt_path, db_path, config.GROUP_COLORS)
        self.message.emit(f"Connection records this session:\n"
                          f"   text:   {txt_path}\n"
                          f"   sqlite: {db_path}")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # --- State carried between frames (same names as main.py) ---
        homography_locked = False
        homography_matrix = None
        homography_size = None

        accumulated_centers = {}
        seen_unexpected = set()

        components = []
        components_detected = False
        last_detection_counts = None
        terminals = []

        video_writer = None
        current_video_path = None
        record_start_time = 0.0
        frames_written = 0
        record_size = (int(actual_width * config.RECORD_SCALE),
                       int(actual_height * config.RECORD_SCALE))

        show_terminals = True

        labjack_continuity = False
        last_labjack_poll = 0.0
        last_reconnect_attempt = 0.0
        candidate_pair = None
        candidate_since = 0.0
        committed_this_touch = False

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        while self._running:
            success, frame = camera.read()
            if not success:
                self.message.emit("Lost the camera. Stopping.")
                break

            operator_frame = frame
            now = time.time()
            self._latest_operator_frame = operator_frame   # for click-sampling

            # --- Stage 2: run YOLO every frame until the FULL expected set of
            # devices appears in ONE frame, then lock (mirrors the ArUco
            # accumulate-then-lock). The board is static, so once a clean frame
            # shows everything, that frame is trustworthy. ---
            if detection_on and not components_detected:
                components = main.detect_components_once(model, operator_frame)
                counts = od.count_detections_by_class(components)

                if expected_counts is None:
                    # No expected set configured -> old behavior: lock at once.
                    components_detected = True
                    terminals = tm.apply_template(components, template_devices)
                    self.message.emit(f"Detected components: {counts}")
                    self.message.emit(f"Stamped {len(terminals)} terminals.")
                else:
                    if counts != last_detection_counts:
                        self.message.emit(
                            f"Detection: have {dict(sorted(counts.items()))}  "
                            f"need {dict(sorted(expected_counts.items()))}")
                        last_detection_counts = counts
                    if counts == expected_counts:
                        components_detected = True
                        terminals = tm.apply_template(components,
                                                      template_devices)
                        self.message.emit("Components locked (full set found). "
                                          f"Stamped {len(terminals)} terminals.")

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
                missing = sorted(config.EXPECTED_MARKER_IDS
                                 - set(accumulated_centers))
                if not missing:
                    homography_matrix, homography_size = main.compute_homography(
                        list(accumulated_centers.values()))
                    homography_locked = True
                    self.message.emit("Homography locked.")

            # --- Stage 1: track the probes on the RAW feed (always) ---
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
            if now - last_labjack_poll >= config.LABJACK_POLL_INTERVAL:
                last_labjack_poll = now
                reading = labjack.read_continuity()        # True / False / None
                labjack_continuity = (reading is True)
                if (not labjack.connected and now - last_reconnect_attempt
                        >= config.LABJACK_RECONNECT_INTERVAL):
                    last_reconnect_attempt = now
                    labjack.try_open()

            red_hit = terminal_hits[0] if len(terminal_hits) > 0 else None
            black_hit = terminal_hits[1] if len(terminal_hits) > 1 else None
            current_pair = None
            if (labjack.connected and labjack_continuity
                    and red_hit is not None and black_hit is not None
                    and red_hit["terminal_id"] != black_hit["terminal_id"]):
                a, b = sorted([red_hit["terminal_id"], black_hit["terminal_id"]])
                current_pair = (a, b)

            if current_pair != candidate_pair:
                candidate_pair = current_pair
                candidate_since = now
                committed_this_touch = False

            if (candidate_pair is not None and not committed_this_touch
                    and now - candidate_since
                    >= config.CONNECTION_DEBOUNCE_SECONDS):
                was_new = log.add_connection(candidate_pair[0],
                                             candidate_pair[1])
                committed_this_touch = True
                if was_new:
                    self.message.emit(f"Connection recorded: "
                                      f"{candidate_pair[0]} <-> "
                                      f"{candidate_pair[1]}")
                else:
                    self.message.emit(f"Connection already on record: "
                                      f"{candidate_pair[0]} <-> "
                                      f"{candidate_pair[1]}")

            # --- Build the WORKING view (board overlays only; status + sampling
            # info go to the side panel instead of onto the video) ---
            working = operator_frame.copy()
            if detection_on and components:
                od.draw_detections(working, components, config.SHOW_LABELS)
            for record in records:
                pt.draw_probe(working, record)
            if show_terminals:
                main.draw_terminals_clean(working, terminals, hit_ids)
                main.draw_group_flags(working, terminals,
                                      log.group_for_terminal)
            active_names = [h["terminal_id"] for h in terminal_hits if h]
            on_text = "On: " + (", ".join(active_names) if active_names else "-")
            main.draw_text_with_outline(working, on_text, (20, 60),
                                        (0, 255, 255), scale=0.7)
            if not labjack.connected:
                main.draw_labjack_warning(working)

            # --- Build the RECTIFIED record view (same as main.py) ---
            rect_annotated = None
            if config.ENABLE_RECTIFICATION and homography_locked:
                rect_annotated = cv2.warpPerspective(
                    operator_frame, homography_matrix, homography_size)
                if detection_on and components:
                    main.draw_components_rectified(rect_annotated, components,
                                                   homography_matrix,
                                                   config.SHOW_LABELS)
                main.draw_probe_tips_rectified(rect_annotated, records,
                                               homography_matrix)
                if show_terminals:
                    main.draw_terminals_clean_rectified(rect_annotated, terminals,
                                                        hit_ids, homography_matrix)
                    main.draw_group_flags_rectified(rect_annotated, terminals,
                                                    log.group_for_terminal,
                                                    homography_matrix)
                if not labjack.connected:
                    main.draw_labjack_warning(rect_annotated)

            # --- Record the working view, PACED BY THE WALL CLOCK (Fix 2) ---
            if video_writer is not None:
                resized = cv2.resize(working, record_size)
                target_frames = int((now - record_start_time) * record_fps)
                while frames_written < target_frames:
                    video_writer.write(resized)
                    frames_written += 1

            # --- Send the finished frames to the GUI for display ---
            display_main = fit_width(working, self.main_view_width)
            self.main_frame_ready.emit(display_main)
            if rect_annotated is not None:
                self.rect_frame_ready.emit(fit_width(rect_annotated, 640))

            # --- Send the status for the side panel ---
            if not config.ENABLE_RECTIFICATION:
                homography_status = "off"
            elif homography_locked:
                homography_status = "locked"
            else:
                have = len(accumulated_centers)
                missing = sorted(config.EXPECTED_MARKER_IDS
                                 - set(accumulated_centers))
                homography_status = f"{have}/4 (missing {missing})"

            if not detection_on:
                components_status = "off"
            elif components_detected:
                components_status = "locked"
            elif expected_counts is not None and last_detection_counts is not None:
                components_status = (f"searching {dict(sorted(last_detection_counts.items()))}")
            else:
                components_status = "searching"

            self.status_ready.emit({
                "probes": [{"label": r["label"], "found": r["found"]}
                           for r in records],
                "homography": homography_status,
                "components": components_status,
                "terminals": len(terminals),
                "labjack": "connected" if labjack.connected else "NOT CONNECTED",
                "connections": len(log.connections),
                "groups": len(log.groups),
                "on_text": on_text,
                "last_sample_bgr": self._last_sample,
                "recording": self._recording,
            })

            # --- Read the mailbox: act on any buttons/clicks since last frame.
            # This is the GUI's version of main.py's key handling, and it sits
            # at the SAME point in the loop so commands act on the freshly built
            # frames (important for Save and Quit). ---
            quit_now = False
            while True:
                try:
                    command = self._commands.get_nowait()
                except queue.Empty:
                    break
                name = command[0]

                if name == "set_active":
                    self._active_idx = command[1]

                elif name == "sample":
                    # command carries (u, v) fractions in 0..1 of the displayed
                    # frame; turn them into raw-frame pixels and sample there.
                    u, v = command[1], command[2]
                    height, width = operator_frame.shape[:2]
                    fx = int(u * width)
                    fy = int(v * height)
                    sample = pt.sample_bgr_from_click(operator_frame, fx, fy)
                    if sample is not None:
                        self._last_sample = sample
                        self.message.emit(f"Sampled BGR={sample} "
                                          f"(Save color to apply).")

                elif name == "save_color":
                    if self._last_sample is None:
                        self.message.emit("No color sampled yet. Click a probe "
                                          "in the video first.")
                    else:
                        probe = config.PROBES[self._active_idx]
                        probe["seed_bgr"] = self._last_sample
                        probe["prev_center"] = None
                        pt.recompute_ranges(probe)
                        self.message.emit(f"Taught {probe['label']} the color "
                                          f"BGR={self._last_sample}")

                elif name == "toggle_terminals":
                    show_terminals = not show_terminals
                    self.message.emit("Terminal circles + group flags: "
                                      f"{'ON' if show_terminals else 'off'}")

                elif name == "save_record":
                    clean_path, legend_path = main.save_records(
                        rect_annotated, working, operator_frame, terminals,
                        log.group_for_terminal, labjack.connected)
                    self.message.emit(f"Saved record: {clean_path}")
                    self.message.emit(f"Saved legend: {legend_path}")

                elif name == "toggle_record":
                    if video_writer is None:
                        path = main.make_timestamped_filename(
                            config.RECORDINGS_FOLDER, "session", "mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(path, fourcc, record_fps,
                                                       record_size)
                        current_video_path = path
                        record_start_time = now
                        frames_written = 0
                        self._recording = True
                        self.message.emit(f"Recording started: {path}")
                    else:
                        video_writer.release()
                        video_writer = None
                        self._recording = False
                        self.message.emit(f"Recording saved: {current_video_path}")

                elif name == "undo":
                    removed = log.undo_last()
                    if removed is not None:
                        self.message.emit(f"Undid connection: "
                                          f"{removed['terminal_a']} <-> "
                                          f"{removed['terminal_b']}")
                        # Block an instant re-record while the probes are still
                        # on the same pair: keep it "already committed" until
                        # they move off it.
                        candidate_pair = current_pair
                        committed_this_touch = True
                    else:
                        self.message.emit("No connections to undo.")

                elif name == "redetect" and detection_on:
                    components_detected = False
                    last_detection_counts = None
                    self.message.emit("Re-running component detection "
                                      "(waiting for the full set)...")

                elif name == "relock" and config.ENABLE_RECTIFICATION:
                    homography_locked = False
                    homography_matrix = None
                    homography_size = None
                    accumulated_centers = {}
                    seen_unexpected = set()
                    self.message.emit("Re-locking: re-finding the ArUco "
                                      "markers from scratch...")

                elif name == "quit":
                    # Save the final record before stopping, just like main.py's
                    # 'q'. We have the freshly built frames in scope right here.
                    clean_path, legend_path = main.save_records(
                        rect_annotated, working, operator_frame, terminals,
                        log.group_for_terminal, labjack.connected)
                    self.message.emit(f"Final record saved: {clean_path}")
                    self.message.emit(f"Final legend saved: {legend_path}")
                    self._running = False
                    quit_now = True
                    break

            if quit_now:
                break

        # --- Cleanup ---
        if video_writer is not None:
            video_writer.release()
            self.message.emit(f"Recording saved: {current_video_path}")
        camera.release()
        self.message.emit("Stopped.")


# ======================================================================
# The window
# ======================================================================

class MainWindow(QMainWindow):
    """The clickable interface: a big raw-video pane, a column of controls and
    status on the right, a dockable rectified-view pane, and an activity log."""

    # How wide (in pixels) the worker downscales each frame to before sending it
    # to us. The video pane then stretches that image to fit its current size,
    # so the video grows with the window. This only affects sharpness when the
    # pane is large, not click accuracy (clicks are mapped by fraction). Raise
    # it for a crisper enlarged picture; lower it if the live feed feels heavy.
    MAIN_VIEW_TRANSPORT_WIDTH = 1280

    # The width of the right-hand control column. Wide enough that no button
    # label is clipped.
    CONTROL_PANEL_WIDTH = 400

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Continuity Annotation System")
        self.resize(1600, 950)              # a roomy starting size; user resizable

        # ---- Central area: video on the left, controls on the right ----
        central = QWidget()
        central_layout = QHBoxLayout(central)

        # The video pane is allowed to EXPAND to fill the window. Each incoming
        # frame is scaled to whatever size the pane currently is, so maximizing
        # the window makes the video bigger. We keep the last frame's pixmap so
        # a click can be mapped back to the underlying frame correctly even when
        # the image is letterboxed inside the pane.
        self._last_main_pixmap = None
        self.video_label = ClickableLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: #202020;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Starting camera...")
        self.video_label.clicked.connect(self.on_video_clicked)
        central_layout.addWidget(self.video_label, stretch=1)

        central_layout.addWidget(self._build_controls_panel())
        self.setCentralWidget(central)

        # ---- The rectified view as a dockable, minimizable pane ----
        self.rect_label = QLabel()
        self.rect_label.setMinimumSize(480, 270)
        self.rect_label.setAlignment(Qt.AlignCenter)
        self.rect_label.setStyleSheet("background-color: #202020;")
        self.rect_label.setText("Rectified view\n(locks once all 4 ArUco "
                                "markers are seen)")

        self.rect_dock = QDockWidget("Rectified record", self)
        self.rect_dock.setWidget(self.rect_label)
        # Allow it to float into its own window (so it can be minimized) or be
        # closed/hidden; it starts docked at the bottom.
        self.rect_dock.setFeatures(QDockWidget.DockWidgetMovable
                                   | QDockWidget.DockWidgetFloatable
                                   | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.rect_dock)

        # ---- The worker thread ----
        self.worker = CameraWorker(self.MAIN_VIEW_TRANSPORT_WIDTH)
        self.worker.main_frame_ready.connect(self.on_main_frame)
        self.worker.rect_frame_ready.connect(self.on_rect_frame)
        self.worker.status_ready.connect(self.on_status)
        self.worker.message.connect(self.on_message)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    # ------------------------------------------------------------------
    # Building the controls + status panel
    # ------------------------------------------------------------------
    def _build_controls_panel(self):
        panel = QWidget()
        panel.setFixedWidth(self.CONTROL_PANEL_WIDTH)
        layout = QVBoxLayout(panel)

        # --- Probe setup ---
        probe_box = QGroupBox("Probe setup")
        probe_layout = QVBoxLayout(probe_box)
        probe_layout.addWidget(QLabel("1) Choose a probe   2) Click it in the "
                                      "video   3) Save color"))

        self.radio_red = QRadioButton(config.PROBES[0]["label"])
        self.radio_black = QRadioButton(config.PROBES[1]["label"])
        self.radio_red.setChecked(True)
        self.radio_red.toggled.connect(
            lambda on: on and self.worker.send(("set_active", 0)))
        self.radio_black.toggled.connect(
            lambda on: on and self.worker.send(("set_active", 1)))
        probe_layout.addWidget(self.radio_red)
        probe_layout.addWidget(self.radio_black)

        swatch_row = QHBoxLayout()
        swatch_row.addWidget(QLabel("Last sampled color:"))
        self.swatch = QLabel()
        self.swatch.setFixedSize(40, 20)
        self.swatch.setStyleSheet("background-color: #000000; border: 1px "
                                  "solid #888;")
        swatch_row.addWidget(self.swatch)
        swatch_row.addStretch()
        probe_layout.addLayout(swatch_row)

        save_color_btn = QPushButton("Save sampled color")
        save_color_btn.clicked.connect(lambda: self.worker.send(("save_color",)))
        probe_layout.addWidget(save_color_btn)
        layout.addWidget(probe_box)

        # --- Records ---
        record_box = QGroupBox("Records")
        record_layout = QVBoxLayout(record_box)
        save_btn = QPushButton("Save record + legend")
        save_btn.clicked.connect(lambda: self.worker.send(("save_record",)))
        self.record_btn = QPushButton("Start recording")
        self.record_btn.clicked.connect(self.on_toggle_record)
        undo_btn = QPushButton("Undo last connection")
        undo_btn.clicked.connect(lambda: self.worker.send(("undo",)))
        record_layout.addWidget(save_btn)
        record_layout.addWidget(self.record_btn)
        record_layout.addWidget(undo_btn)
        layout.addWidget(record_box)

        # --- View + setup ---
        setup_box = QGroupBox("View + setup")
        setup_layout = QVBoxLayout(setup_box)
        terminals_btn = QPushButton("Show / hide terminals")
        terminals_btn.clicked.connect(
            lambda: self.worker.send(("toggle_terminals",)))
        redetect_btn = QPushButton("Re-detect components")
        redetect_btn.clicked.connect(lambda: self.worker.send(("redetect",)))
        relock_btn = QPushButton("Re-lock homography")
        relock_btn.clicked.connect(lambda: self.worker.send(("relock",)))
        rect_btn = QPushButton("Show / hide rectified view")
        rect_btn.clicked.connect(
            lambda: self.rect_dock.setVisible(not self.rect_dock.isVisible()))
        setup_layout.addWidget(terminals_btn)
        setup_layout.addWidget(redetect_btn)
        setup_layout.addWidget(relock_btn)
        setup_layout.addWidget(rect_btn)
        layout.addWidget(setup_box)

        # --- Live status ---
        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout(status_box)
        self.status_labels = {
            "probes": QLabel("Probes: -"),
            "homography": QLabel("Homography: -"),
            "components": QLabel("Components: -"),
            "terminals": QLabel("Terminals: -"),
            "labjack": QLabel("LabJack: -"),
            "connections": QLabel("Connections: -"),
            "groups": QLabel("Groups: -"),
            "on_text": QLabel("On: -"),
            "recording": QLabel(""),
        }
        for label in self.status_labels.values():
            label.setWordWrap(True)
            status_layout.addWidget(label)
        layout.addWidget(status_box)

        # --- Activity log: let it grow to fill the leftover vertical space,
        # so it shows many lines when the window is tall. ---
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(150)
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(QLabel("Activity log"))
        layout.addWidget(self.log_view, stretch=1)

        # --- Quit (stays pinned at the bottom, below the growing log) ---
        quit_btn = QPushButton("Save final record + quit")
        quit_btn.clicked.connect(self.close)
        layout.addWidget(quit_btn)

        return panel

    # ------------------------------------------------------------------
    # Slots: the worker hands us frames / status; we update widgets here, on the
    # GUI thread (the only thread allowed to touch widgets).
    # ------------------------------------------------------------------
    def on_main_frame(self, bgr_image):
        # Remember the full-size pixmap (so a click can be mapped back to it),
        # then show it scaled to the pane's current size. Because new frames
        # stream continuously, resizing the window re-scales the video within a
        # frame or two automatically.
        self._last_main_pixmap = bgr_to_qpixmap(bgr_image)
        scaled = self._last_main_pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def on_rect_frame(self, bgr_image):
        pixmap = bgr_to_qpixmap(bgr_image)
        self.rect_label.setPixmap(pixmap.scaled(
            self.rect_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_status(self, status):
        probe_bits = ", ".join(
            f"{p['label']} {'OK' if p['found'] else '-'}"
            for p in status["probes"])
        self.status_labels["probes"].setText(f"Probes: {probe_bits}")
        self.status_labels["homography"].setText(
            f"Homography: {status['homography']}")
        self.status_labels["components"].setText(
            f"Components: {status['components']}")
        self.status_labels["terminals"].setText(
            f"Terminals: {status['terminals']}")

        labjack_label = self.status_labels["labjack"]
        labjack_label.setText(f"LabJack: {status['labjack']}")
        if status["labjack"] == "connected":
            labjack_label.setStyleSheet("")
        else:
            labjack_label.setStyleSheet("color: red; font-weight: bold;")

        self.status_labels["connections"].setText(
            f"Connections: {status['connections']}")
        self.status_labels["groups"].setText(f"Groups: {status['groups']}")
        self.status_labels["on_text"].setText(status["on_text"])

        recording_label = self.status_labels["recording"]
        if status["recording"]:
            recording_label.setText("● RECORDING")
            recording_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            recording_label.setText("")

        sample = status["last_sample_bgr"]
        if sample is not None:
            b, g, r = sample
            self.swatch.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #888;")

    def on_message(self, text):
        self.log_view.appendPlainText(text)

    def on_failed(self, text):
        QMessageBox.critical(self, "Camera worker error", text)

    # ------------------------------------------------------------------
    # Button helpers
    # ------------------------------------------------------------------
    def on_video_clicked(self, x, y):
        # The video is scaled to fit the pane while keeping its shape, so there
        # may be empty (letterbox) bars on the sides or top/bottom. To sample
        # the right pixel we must (1) find where the video image actually sits
        # inside the pane, (2) reject clicks that land on the bars, and (3) turn
        # the rest into 0..1 fractions of the IMAGE, which the worker maps onto
        # the full-resolution frame.
        if self._last_main_pixmap is None:
            return
        pane_w = self.video_label.width()
        pane_h = self.video_label.height()

        # Size the image is actually drawn at inside the pane.
        shown = self._last_main_pixmap.size().scaled(
            pane_w, pane_h, Qt.KeepAspectRatio)
        shown_w, shown_h = shown.width(), shown.height()
        if shown_w <= 0 or shown_h <= 0:
            return

        # The image is centered, so work out the bar offsets and subtract them.
        offset_x = (pane_w - shown_w) / 2
        offset_y = (pane_h - shown_h) / 2
        inside_x = x - offset_x
        inside_y = y - offset_y
        if inside_x < 0 or inside_y < 0 or inside_x > shown_w or inside_y > shown_h:
            return                         # clicked an empty bar, not the video

        u = min(max(inside_x / shown_w, 0.0), 1.0)
        v = min(max(inside_y / shown_h, 0.0), 1.0)
        self.worker.send(("sample", u, v))

    def on_toggle_record(self):
        self.worker.send(("toggle_record",))
        # Flip the button label optimistically; the status panel shows the
        # authoritative recording state from the worker.
        if self.record_btn.text() == "Start recording":
            self.record_btn.setText("Stop recording")
        else:
            self.record_btn.setText("Start recording")

    # ------------------------------------------------------------------
    # Closing: ask the worker to save the final record and stop, then wait for
    # it to finish so the camera and any video file are released cleanly.
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.send(("quit",))
            if not self.worker.wait(3000):    # up to 3 seconds for a clean stop
                self.worker.request_stop()
                self.worker.wait(1000)
        event.accept()


# NOTE: this entry point is called run_gui(), NOT main(). We must not name it
# main(), because this file does `import main` at the top to reuse main.py's
# helpers - a `def main()` here would overwrite that imported module name and
# break every `main.<helper>` call.
def run_gui():
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()