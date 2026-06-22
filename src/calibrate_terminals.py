# calibrate_terminals.py
#
# ONE-TIME tool that builds the terminal-map template (terminal_map.json).
#
# You calibrate ONE device of each class. The terminals you mark are stored as
# (u, v) positions inside that device's oriented bounding box, so at run time
# main.py can stamp those same terminals onto EVERY device of that class,
# wherever it is placed on the board.
#
# How it works, step by step:
#   1. The tool runs the detector ONCE so it knows where each device's box is.
#   2. You teach the red probe its tape color (click the probe, then 's').
#   3. You pick a device class to calibrate (f / p / j / k).
#   4. You rest the probe on a terminal and press 'x'. The probe's CENTROID is
#      measured against the chosen device's box, turned into (u, v), and stored
#      as the next terminal for that class. A clean colored circle (no text)
#      marks it.
#   5. 'u' undoes the last mark; 'd' deselects the device; pressing the same
#      device key again resumes adding to it.
#   6. 'q' saves everything and quits.
#
# Run this at the SAME camera resolution you run main.py at.
#
# Controls are also shown in the on-screen "Controls" window (toggle with 'h').

import cv2
import numpy as np

import config
import probe_tracking as pt
import terminal_map as tm
import ui

try:
    import object_detection as od
    from ultralytics import YOLO
    DETECTION_AVAILABLE = True
    DETECTION_IMPORT_ERROR = None
except Exception as error:                # pragma: no cover - machine dependent
    od = None
    YOLO = None
    DETECTION_AVAILABLE = False
    DETECTION_IMPORT_ERROR = str(error)


CONTROLS_TITLE = "Calibration controls"
CONTROLS_SECTIONS = [
    ("Set up the probe", [
        ("click + s", "teach the red probe its color"),
    ]),
    ("Pick a device", [
        ("f", "Flathead_Block"),
        ("p", "Phillips_Block"),
        ("j", "Terminal_1"),
        ("k", "Terminal_2"),
        ("d", "deselect current device"),
    ]),
    ("Mark terminals", [
        ("x", "mark terminal at probe centroid"),
        ("u", "undo the last mark"),
        ("o", "re-run detection"),
    ]),
    ("Window", [
        ("h", "show / hide this controls window"),
        ("q", "finish and save"),
    ]),
]


def detect_once(model, frame):
    """Run YOLO a single time and return detection records in operator-frame
    pixels, duplicates removed. Mirrors main.detect_components_once()."""
    height, width = frame.shape[:2]
    if config.CAMERA_FLIP_CODE is None:
        model_frame = frame
    else:
        model_frame = cv2.flip(frame, config.CAMERA_FLIP_CODE)
    results = model.predict(model_frame, conf=config.CONFIDENCE_THRESHOLD,
                            verbose=False)
    records = od.extract_detections(results[0], model.names,
                                    config.CAMERA_FLIP_CODE, width, height)
    return od.remove_overlapping_duplicates(records, config.OVERLAP_THRESHOLD)


def nearest_box_of_class(detections, class_name, point):
    """Return the detection of the given class whose box center is closest to
    'point' (or None if no box of that class was detected)."""
    candidates = [d for d in detections if d["class_name"] == class_name]
    if not candidates:
        return None

    def distance(detection):
        cx, cy = tm.box_center(detection["corners"])
        return (cx - point[0]) ** 2 + (cy - point[1]) ** 2

    return min(candidates, key=distance)


def main():
    if not DETECTION_AVAILABLE:
        print("The detector could not be loaded, so calibration cannot run:")
        print(f"   {DETECTION_IMPORT_ERROR}")
        return

    # --- Map class name <-> color, and class name <-> the key that selects it ---
    device_for_key = config.CALIBRATION_DEVICE_KEYS    # {"f": "Flathead_Block", ...}

    # --- Load the model ---
    try:
        print("Loading model...")
        model = YOLO(config.MODEL_PATH)
        print("Model loaded.")
    except Exception as error:
        print(f"Could not load model '{config.MODEL_PATH}': {error}")
        return

    class_index_of = {name: index for index, name in model.names.items()}

    def class_color(class_name):
        index = class_index_of.get(class_name, 0)
        return tm.color_for_class(index, od.BOX_COLORS)

    # --- Open the camera (same settings as main.py) ---
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open camera at index {config.CAMERA_INDEX}.")
        return
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Prepare the red probe (probe 0) ---
    for probe in config.PROBES:
        probe["prev_center"] = None
        pt.recompute_ranges(probe)
    red_probe = config.PROBES[0]

    # --- One-time detection ---
    success, frame = camera.read()
    if not success:
        print("Could not read from the camera.")
        camera.release()
        return
    detections = detect_once(model, frame)
    print("Detected:", od.count_detections_by_class(detections))

    # --- Template being built ---
    # devices[class_name] = {"calibration_box": [[x,y]*4] or None,
    #                        "terminals": [{"index","u","v","calib_xy"}, ...]}
    devices = {}
    capture_order = []          # list of class names, in the order marks were made

    # --- State the mouse callback shares ---
    state = {"surface": None, "last_sample": None, "active_class": None}

    def on_mouse(event, x, y, flags, param):
        # In this tool a click only ever SAMPLES the probe color. Terminals are
        # placed with 'x', never by clicking, so the two can't be confused.
        if event != cv2.EVENT_LBUTTONDOWN or state["surface"] is None:
            return
        sample = pt.sample_bgr_from_click(state["surface"], x, y)
        if sample is None:
            return
        state["last_sample"] = sample
        print(f"[click] sampled BGR={sample} (press 's' to teach the red probe)")

    WINDOW = "Calibration"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    cv2.setMouseCallback(WINDOW, on_mouse)

    controls_open = True
    ui.show_controls_window("Controls", CONTROLS_TITLE, CONTROLS_SECTIONS)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    print("Ready. Teach the probe color, pick a device (f/p/j/k), then 'x'.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        success, frame = camera.read()
        if not success:
            print("Lost the camera. Exiting.")
            break

        operator_frame = frame
        state["surface"] = operator_frame.copy()

        # Track the red probe so we can read its centroid when 'x' is pressed.
        hsv = cv2.cvtColor(operator_frame, cv2.COLOR_BGR2HSV)
        record = pt.track_probe(hsv, red_probe, operator_frame.shape, kernel,
                                config.PROBE_MAX_JUMP_FRACTION)

        view = operator_frame.copy()

        # Draw each detected device's box, faint, in its class color.
        for detection in detections:
            color = tm.color_for_class(detection["class_index"], od.BOX_COLORS)
            points = np.array(detection["corners"], dtype=np.int32)
            cv2.polylines(view, [points], isClosed=True, color=color, thickness=2)

        # Redraw every terminal marked so far, by re-applying its (u, v) to the
        # locked calibration box (this also confirms the round trip is correct).
        for class_name, device in devices.items():
            if not device["calibration_box"]:
                continue
            ordered = np.array(device["calibration_box"], dtype=np.float32)
            color = class_color(class_name)
            for terminal in device["terminals"]:
                xy = tm.uv_to_point(ordered, terminal["u"], terminal["v"])
                tm.draw_terminal_circle(view, xy, color)

        # Draw the probe itself.
        pt.draw_probe(view, record)

        # Status text in the corners (text here is fine; it never lands on the
        # final record image). Black outline then bright text for readability.
        active = state["active_class"] or "(none - press f/p/j/k)"
        marked_active = len(devices.get(state["active_class"], {}).get("terminals", [])) \
            if state["active_class"] else 0
        total = sum(len(d["terminals"]) for d in devices.values())
        lines = [
            f"Calibrating: {active}   marks on this device: {marked_active}",
            f"Total terminals marked: {total}",
        ]
        for i, text in enumerate(lines):
            y = 36 + i * 34
            cv2.putText(view, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF

        if key == 255:                   # no key pressed this frame
            continue

        if key == ord('q'):
            break

        elif key == ord('h'):
            controls_open = not controls_open
            if controls_open:
                ui.show_controls_window("Controls", CONTROLS_TITLE, CONTROLS_SECTIONS)
            else:
                cv2.destroyWindow("Controls")

        elif key == ord('s'):
            if state["last_sample"] is None:
                print("No color sampled yet. Click on the probe first.")
            else:
                red_probe["seed_bgr"] = state["last_sample"]
                red_probe["prev_center"] = None
                pt.recompute_ranges(red_probe)
                print(f"Taught the red probe BGR={state['last_sample']}")

        elif key == ord('o'):
            detections = detect_once(model, operator_frame)
            print("Re-detected:", od.count_detections_by_class(detections))

        elif chr(key) in device_for_key:
            state["active_class"] = device_for_key[chr(key)]
            print(f"Calibrating: {state['active_class']}")

        elif key == ord('d'):
            state["active_class"] = None
            print("Deselected. Press f/p/j/k to pick a device again.")

        elif key == ord('x'):
            class_name = state["active_class"]
            if class_name is None:
                print("Pick a device first (f/p/j/k).")
            elif not record["found"]:
                print("The probe is not detected - cannot mark a terminal.")
            else:
                centroid = record["center"]
                device = devices.setdefault(
                    class_name, {"calibration_box": None, "terminals": []})

                # Lock onto the calibration box the first time we mark this class.
                if device["calibration_box"] is None:
                    box = nearest_box_of_class(detections, class_name, centroid)
                    if box is None:
                        print(f"No '{class_name}' box detected. Press 'o' to "
                              f"re-detect, then try again.")
                        continue
                    ordered = tm.order_box(box["corners"])
                    device["calibration_box"] = ordered.tolist()

                ordered = np.array(device["calibration_box"], dtype=np.float32)
                u, v = tm.point_to_uv(ordered, centroid)
                index = len(device["terminals"]) + 1
                device["terminals"].append({
                    "index": index, "u": u, "v": v,
                    "calib_xy": [float(centroid[0]), float(centroid[1])],
                })
                capture_order.append(class_name)
                print(f"Marked {class_name}_{index}  (u={u:.3f}, v={v:.3f})")

        elif key == ord('u'):
            if capture_order:
                class_name = capture_order.pop()
                removed = devices[class_name]["terminals"].pop()
                print(f"Removed {class_name}_{removed['index']}")
            else:
                print("Nothing to undo.")

    # --- Save and quit ---
    if devices:
        tm.save_template(config.TERMINAL_MAP_PATH, devices,
                         image_size=[actual_width, actual_height],
                         note="One device per class; terminals stored as (u,v).")
        total = sum(len(d["terminals"]) for d in devices.values())
        print(f"Saved {total} terminals across {len(devices)} device classes "
              f"to {config.TERMINAL_MAP_PATH}")
    else:
        print("Nothing was marked, so no file was written.")

    camera.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()