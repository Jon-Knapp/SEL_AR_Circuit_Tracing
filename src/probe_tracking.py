# probe_tracking.py
#
# Color-based tracking of the two test probes, as a set of importable
# functions (the same style as object_detection.py). main.py imports these.
#
# The idea, in plain terms:
#   Each probe is a distinct color (the red probe's red body, and the neon
#   magenta tape on the other probe). For each video frame we:
#     1. Convert the frame to HSV. HSV separates "what color" (Hue) from
#        "how vivid" (Saturation) and "how bright" (Value), which makes
#        color matching far more reliable than raw camera (BGR) values.
#     2. Build a black-and-white MASK: white where the pixel matches a
#        probe's color range, black everywhere else.
#     3. Clean the mask of speckle noise.
#     4. Pick the single best white blob (biggest, and near where the probe
#        was last frame, so a far-off same-color object can't steal it).
#     5. Estimate the contact point (the "tip") from that blob.
#
# You do NOT hand-tune the color numbers. In main.py you CLICK on each probe
# in the live video to sample its color; this module turns that sampled
# color into the HSV range automatically (see compute_hsv_ranges_from_bgr).

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Turning a clicked color into an HSV range
# ----------------------------------------------------------------------

def compute_hsv_ranges_from_bgr(bgr, h_tol, s_tol, v_tol, s_min, v_min):
    """
    Convert ONE sampled color (in B,G,R order) into one or two HSV ranges.

    Why one OR two ranges? In OpenCV the Hue wheel is 0–179 and wraps around:
    red lives at BOTH ends (near 0 and near 179). If the color sits near an
    end, a single range can't cover both sides of the wrap, so we return two.

    Each range is a (lower, upper) pair of np.array([H, S, V]).

      h_tol      : how far in Hue (color) we still accept around the sample
      s_tol/v_tol: same idea for Saturation and Value
      s_min/v_min: floors. Below these the pixel is nearly grey/black, where
                   Hue is meaningless, so we refuse to accept that low.
    """
    # Convert the single BGR sample to HSV.
    bgr_pixel = np.uint8([[bgr]])                      # shape (1, 1, 3)
    h, s, v = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(h), int(s), int(v)

    h_low, h_high = h - h_tol, h + h_tol
    s_low, s_high = max(s_min, s - s_tol), min(255, s + s_tol)
    v_low, v_high = max(v_min, v - v_tol), min(255, v + v_tol)

    ranges = []

    if h_low < 0:
        # Wrapped past 0: cover [0 .. h_high] AND [180+h_low .. 179].
        ranges.append((np.array([0, s_low, v_low], np.uint8),
                       np.array([h_high, s_high, v_high], np.uint8)))
        ranges.append((np.array([180 + h_low, s_low, v_low], np.uint8),
                       np.array([179, s_high, v_high], np.uint8)))
    elif h_high > 179:
        # Wrapped past 179: cover [h_low .. 179] AND [0 .. h_high-180].
        ranges.append((np.array([h_low, s_low, v_low], np.uint8),
                       np.array([179, s_high, v_high], np.uint8)))
        ranges.append((np.array([0, s_low, v_low], np.uint8),
                       np.array([h_high - 180, s_high, v_high], np.uint8)))
    else:
        # No wrap: a single range is enough.
        ranges.append((np.array([h_low, s_low, v_low], np.uint8),
                       np.array([h_high, s_high, v_high], np.uint8)))

    return ranges


def recompute_ranges(probe):
    """
    Refresh probe["hsv_ranges"] from probe["seed_bgr"] and the probe's
    tolerance settings. Call this once at startup and again whenever the
    user clicks-and-saves a new color sample for this probe.
    """
    probe["hsv_ranges"] = compute_hsv_ranges_from_bgr(
        probe["seed_bgr"],
        h_tol=probe["h_tol"], s_tol=probe["s_tol"], v_tol=probe["v_tol"],
        s_min=probe["s_min"], v_min=probe["v_min"],
    )
    hsv = cv2.cvtColor(np.uint8([[probe["seed_bgr"]]]), cv2.COLOR_BGR2HSV)[0, 0]
    print(f"[{probe['label']}] seed BGR={tuple(probe['seed_bgr'])} "
          f"HSV={tuple(int(c) for c in hsv)}")


# ----------------------------------------------------------------------
# Building and cleaning the color mask
# ----------------------------------------------------------------------

def build_color_mask(hsv_frame, hsv_ranges):
    """
    Return a white-on-black mask: white where the pixel falls inside ANY of
    the given HSV ranges. We OR the ranges together so red's two-piece range
    becomes one mask.
    """
    mask = None
    for lower, upper in hsv_ranges:
        piece = cv2.inRange(hsv_frame, lower, upper)
        mask = piece if mask is None else cv2.bitwise_or(mask, piece)
    return mask


def clean_mask(mask, open_iter, close_iter, kernel):
    """
    Remove noise from the mask.
      - medianBlur knocks out salt-and-pepper specks.
      - MORPH_OPEN  (erode then dilate) deletes small stray blobs.
      - MORPH_CLOSE (dilate then erode) fills small holes inside a blob.
    More iterations = cleaner, but too many can erase a small real blob.
    """
    clean = cv2.medianBlur(mask, 5)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kernel, iterations=open_iter)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return clean


# ----------------------------------------------------------------------
# Picking the best blob
# ----------------------------------------------------------------------

def find_best_blob(mask, min_area, max_area, prev_center, max_dist):
    """
    From all white blobs in the mask, choose the single one that is the
    probe. Returns (bbox, center, blob_mask) or (None, None, None).

      bbox      : (x, y, width, height)
      center    : (cx, cy) integer center of mass
      blob_mask : a mask containing ONLY the chosen blob (used for tip math)

    A blob qualifies only if its area is within [min_area, max_area]. If we
    knew where the probe was last frame (prev_center), we also reject blobs
    that are further than max_dist away (a probe can't jump that far in one
    frame), and we prefer blobs that are both BIG and CLOSE.
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    best_index = -1
    best_score = -1e18

    for i in range(1, num):                    # i = 0 is the black background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        cx, cy = centroids[i]
        if prev_center is not None:
            distance = np.hypot(cx - prev_center[0], cy - prev_center[1])
            if distance > max_dist:
                continue
            score = area - 0.5 * distance      # reward big AND close
        else:
            score = area                       # first frame: just biggest

        if score > best_score:
            best_score = score
            best_index = i

    if best_index == -1:
        return None, None, None

    x = stats[best_index, cv2.CC_STAT_LEFT]
    y = stats[best_index, cv2.CC_STAT_TOP]
    w = stats[best_index, cv2.CC_STAT_WIDTH]
    h = stats[best_index, cv2.CC_STAT_HEIGHT]
    cx, cy = centroids[best_index]

    blob_mask = np.uint8(labels == best_index) * 255
    return (x, y, w, h), (int(cx), int(cy)), blob_mask


# ----------------------------------------------------------------------
# Estimating the contact point (the tip)
# ----------------------------------------------------------------------

def estimate_tip(center, blob_mask, method, frame_shape):
    """
    Estimate where the probe touches the board.

    method = "centroid":
        Return the blob's center of mass. This is best when the colored
        region is SMALL and sits NEAR the metal tip (e.g. a short piece of
        tape), because then the center is already almost at the tip.

    method = "axis_end":
        Treat the blob as an elongated shape, find its long axis with
        cv2.minAreaRect, and return the END of that axis that is FARTHER
        from the nearest frame border. The reasoning: the operator's hand
        enters from a frame edge, so the hand side of the probe is near an
        edge and the tip points inward. Better for a LONG colored body.

        Honest caveat: this assumes the hand enters from an edge and the
        tip points inward. If the operator reaches so their hand is near
        the frame center with the tip toward an edge, this can flip and
        report the wrong end. For a short piece of tip-tape, prefer
        "centroid" — it has no such failure mode.
    """
    if method == "centroid" or blob_mask is None:
        return center

    ys, xs = np.where(blob_mask > 0)
    if len(xs) < 5:
        return center                          # too few pixels to fit an axis

    points = np.column_stack([xs, ys]).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(points))   # four corners of the blob

    # The long axis joins the midpoints of the two SHORT sides. Compute both
    # candidate axes (the two ways to pair opposite sides) and keep the longer.
    mid_01 = (box[0] + box[1]) / 2
    mid_12 = (box[1] + box[2]) / 2
    mid_23 = (box[2] + box[3]) / 2
    mid_30 = (box[3] + box[0]) / 2

    axis_a = (mid_01, mid_23)
    axis_b = (mid_12, mid_30)
    if np.linalg.norm(mid_01 - mid_23) >= np.linalg.norm(mid_12 - mid_30):
        ends = axis_a
    else:
        ends = axis_b

    height, width = frame_shape[:2]

    def distance_to_nearest_border(point):
        x, y = point
        return min(x, y, width - 1 - x, height - 1 - y)

    tip = max(ends, key=distance_to_nearest_border)
    return (int(tip[0]), int(tip[1]))


# ----------------------------------------------------------------------
# Tracking one probe for one frame
# ----------------------------------------------------------------------

def track_probe(hsv_frame, probe, frame_shape, kernel, max_jump_fraction):
    """
    Run the full pipeline for ONE probe on ONE frame and return a record:

        {
          "label": ..., "found": bool, "draw_color": (B,G,R),
          "bbox": (x,y,w,h) or None, "center": (cx,cy) or None,
          "tip": (x,y) or None, "raw_mask": ndarray, "clean_mask": ndarray,
        }

    Side effect: updates probe["prev_center"] so the next frame can use it.
    """
    raw = build_color_mask(hsv_frame, probe["hsv_ranges"])
    clean = clean_mask(raw, probe["open_iter"], probe["close_iter"], kernel)

    max_dist = int(max_jump_fraction * frame_shape[1])    # fraction of width
    bbox, center, blob_mask = find_best_blob(
        clean, probe["min_area"], probe["max_area"],
        probe.get("prev_center"), max_dist)

    record = {
        "label": probe["label"],
        "draw_color": probe["draw_color"],
        "raw_mask": raw,
        "clean_mask": clean,
    }

    if bbox is None:
        # Lost this frame. Forget the last position so we can re-acquire the
        # probe anywhere next frame instead of staying anchored to thin air.
        probe["prev_center"] = None
        record.update(found=False, bbox=None, center=None, tip=None)
        return record

    probe["prev_center"] = center
    tip = estimate_tip(center, blob_mask, probe.get("tip_method", "centroid"),
                       frame_shape)
    record.update(found=True, bbox=bbox, center=center, tip=tip)
    return record


# ----------------------------------------------------------------------
# Drawing + click sampling helpers
# ----------------------------------------------------------------------

def draw_probe(frame, record):
    """Draw one probe's box, tip marker, and label. No-op if not found."""
    if not record["found"]:
        return

    color = record["draw_color"]
    x, y, w, h = record["bbox"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    tip_x, tip_y = record["tip"]
    cv2.circle(frame, (tip_x, tip_y), 12, color, 2)     # ring
    cv2.circle(frame, (tip_x, tip_y),  4, color, -1)    # center dot

    label = record["label"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, 0.6, 2)
    label_y = max(text_h + 4, y - 8)
    cv2.rectangle(frame, (x, label_y - text_h - baseline),
                  (x + text_w + 4, label_y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, label, (x + 2, label_y), font, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


def sample_bgr_from_click(frame, x, y):
    """
    Read the color at a clicked pixel. We take the MEDIAN of a small 5x5
    patch instead of the single pixel, so one stray pixel can't throw off
    the sample. Returns (B, G, R) ints, or None if the click was off-frame.
    """
    patch = frame[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    if patch.size == 0:
        return None
    bgr = np.median(patch.reshape(-1, 3), axis=0).astype(int)
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))