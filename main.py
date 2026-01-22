import cv2


def clip_box(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w - 1))
    y2 = max(y1 + 1, min(y2, h - 1))
    return x1, y1, x2, y2


def draw_box(frame, box, color=(0, 255, 0), thickness=2):
    h, w = frame.shape[:2]
    box = clip_box(box, w, h)
    cv2.rectangle(frame, box[:2], box[2:], color, thickness)
    return box


def draw_label(frame, box, text, bg_color, scale=0.5, thickness=1, pad=3):
    h, w = frame.shape[:2]
    x1, y1, _, _ = box

    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    lw, lh = tw + 2 * pad, th + base + 2 * pad
    ly1 = y1 - lh if y1 - lh > 0 else y1 + 2
    lx1 = min(max(0, x1), w - lw)

    b, g, r = bg_color
    text_color = (
        (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)
    )

    cv2.rectangle(frame, (lx1, ly1), (lx1 + lw, ly1 + lh), bg_color, -1)
    cv2.putText(
        frame,
        text,
        (lx1 + pad, ly1 + lh - base - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_box_with_label(frame, box, text=None, color=(0, 255, 0)):
    box = draw_box(frame, box, color)
    if text:
        draw_label(frame, box, text, color)
    return box


frame = cv2.imread("frame.jpg")

bbox = (50, 60, 220, 200)
score = 0.93

draw_box_with_label(frame, bbox, f"Person {score:.2f}", color=(0, 128, 255))

cv2.imshow("img", frame)
cv2.waitKey(0)
