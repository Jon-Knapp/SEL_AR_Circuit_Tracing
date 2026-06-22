# ui.py
#
# A small shared helper that draws an on-screen "Controls" window, so you don't
# have to remember every key. Both the calibration tool and main.py use it;
# each one passes in its OWN list of commands, organized into sections.
#
# A "section" is a (heading, items) pair, where items is a list of
# (key, description) pairs. For example:
#
#   sections = [
#       ("Probes", [("1 / 2", "choose probe"), ("click + s", "teach color")]),
#       ("Records", [("c", "save image"), ("r", "record")]),
#   ]

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_controls_image(title, sections, width=560):
    """Build a dark image listing the title and every command, and return it."""
    line_height = 28
    pad = 20

    # Count how many text lines we need so the image is exactly tall enough.
    line_count = 2                       # title + one blank line under it
    for heading, items in sections:
        line_count += 1                  # the section heading
        line_count += len(items)         # one line per command
        line_count += 1                  # a blank line after the section

    height = pad * 2 + line_height * line_count
    image = np.full((height, width, 3), 30, dtype=np.uint8)   # dark grey

    y = pad + line_height
    cv2.putText(image, title, (pad, y), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    y += line_height * 2

    for heading, items in sections:
        cv2.putText(image, heading, (pad, y), FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        y += line_height
        for key, description in items:
            cv2.putText(image, f"[{key}]", (pad + 12, y), FONT, 0.5,
                        (120, 220, 120), 1, cv2.LINE_AA)
            cv2.putText(image, description, (pad + 150, y), FONT, 0.5,
                        (220, 220, 220), 1, cv2.LINE_AA)
            y += line_height
        y += line_height                 # blank line between sections

    return image


def show_controls_window(window_name, title, sections):
    """Open (or refresh) the controls window. The content is static, so you can
    call this once when you want it shown."""
    image = make_controls_image(title, sections)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
    cv2.imshow(window_name, image)