# make_charuco_board.py
#
# Generates the printable 7 x 10 ChArUco calibration board as a PNG, sized so
# that if you print it at 100% scale the squares come out the physical size
# named in camera_config.py.
#
# Run it once:
#     python make_charuco_board.py
#
# Then:
#   1. Print charuco_board.png at 100% / "Actual size". Turn OFF "fit to page",
#      "shrink oversized pages", and any "scale to fit" option. These are on by
#      default in most print dialogs and they will silently shrink the board.
#   2. Glue or tape the sheet to something RIGID and FLAT - a clipboard, a
#      piece of hardboard, foam board. A calibration board that bows even a
#      millimetre puts a systematic error into every measurement you take
#      afterwards, and nothing downstream can detect or undo it.
#   3. MEASURE the printed squares. Lay a steel rule across five squares,
#      read the total, divide by five, and put that number into
#      CHARUCO_SQUARE_LENGTH_MM in camera_config.py. Do the same for one
#      marker's black square and CHARUCO_MARKER_LENGTH_MM.
#
# Step 3 is not optional. Everything the pose tool later reports in millimetres
# is scaled by these two numbers.

import os

import cv2
import numpy as np

import camera_config as cfg


def millimetres_to_pixels(millimetres, dots_per_inch):
    """Convert a physical length to a pixel count at a given print resolution.
    There are exactly 25.4 millimetres in an inch."""
    return int(round(millimetres * dots_per_inch / 25.4))


def main():
    dictionary = cv2.aruco.getPredefinedDictionary(cfg.CHARUCO_DICTIONARY)

    # --- Sanity check: does this board need more markers than the dictionary
    #     actually contains? Catch it here rather than halfway through a print.
    squares_total = cfg.CHARUCO_SQUARES_X * cfg.CHARUCO_SQUARES_Y
    markers_needed = squares_total // 2
    markers_available = len(dictionary.bytesList)
    if markers_needed > markers_available:
        print(f"This {cfg.CHARUCO_SQUARES_X} x {cfg.CHARUCO_SQUARES_Y} board "
              f"needs {markers_needed} markers, but the chosen dictionary only "
              f"has {markers_available}.")
        print("Use a smaller board or a dictionary with more markers.")
        return

    # --- Sanity check: the marker must be smaller than the square it sits in.
    if cfg.CHARUCO_MARKER_LENGTH_MM >= cfg.CHARUCO_SQUARE_LENGTH_MM:
        print("CHARUCO_MARKER_LENGTH_MM must be smaller than "
              "CHARUCO_SQUARE_LENGTH_MM.")
        print("A ratio of about 0.7 to 0.8 works well - the white gap around "
              "each marker is what lets the detector find it.")
        return

    ratio = cfg.CHARUCO_MARKER_LENGTH_MM / cfg.CHARUCO_SQUARE_LENGTH_MM
    if not (0.6 <= ratio <= 0.85):
        print(f"Warning: marker/square ratio is {ratio:.2f}. Between 0.7 and "
              f"0.8 is the usual advice. Carrying on anyway.")

    # --- Build the board definition.
    #
    # cv2.aruco.CharucoBoard takes the size as (squares across, squares down).
    # The two lengths can be given in any consistent unit; we use millimetres,
    # which is why every distance the pose tool reports later is in millimetres.
    board = cv2.aruco.CharucoBoard(
        (cfg.CHARUCO_SQUARES_X, cfg.CHARUCO_SQUARES_Y),
        cfg.CHARUCO_SQUARE_LENGTH_MM,
        cfg.CHARUCO_MARKER_LENGTH_MM,
        dictionary,
    )

    # --- Work out the image size in pixels so the print comes out to scale.
    board_width_mm = cfg.CHARUCO_SQUARES_X * cfg.CHARUCO_SQUARE_LENGTH_MM
    board_height_mm = cfg.CHARUCO_SQUARES_Y * cfg.CHARUCO_SQUARE_LENGTH_MM

    width_px = millimetres_to_pixels(board_width_mm, cfg.PRINT_DPI)
    height_px = millimetres_to_pixels(board_height_mm, cfg.PRINT_DPI)
    margin_px = millimetres_to_pixels(cfg.PRINT_MARGIN_MM, cfg.PRINT_DPI)

    image = board.generateImage((width_px, height_px), marginSize=margin_px,
                                borderBits=1)

    output_path = "charuco_board.png"
    cv2.imwrite(output_path, image)

    # --- Tell the user exactly what they are holding.
    total_width_mm = board_width_mm + 2 * cfg.PRINT_MARGIN_MM
    total_height_mm = board_height_mm + 2 * cfg.PRINT_MARGIN_MM

    print(f"Wrote {output_path}")
    print()
    print(f"  Board:         {cfg.CHARUCO_SQUARES_X} x {cfg.CHARUCO_SQUARES_Y} "
          f"squares, {markers_needed} markers")
    print(f"  Square size:   {cfg.CHARUCO_SQUARE_LENGTH_MM} mm")
    print(f"  Marker size:   {cfg.CHARUCO_MARKER_LENGTH_MM} mm")
    print(f"  Printed size:  {total_width_mm:.0f} x {total_height_mm:.0f} mm "
          f"(including the {cfg.PRINT_MARGIN_MM:.0f} mm margin)")
    print(f"  Image:         {image.shape[1]} x {image.shape[0]} px "
          f"at {cfg.PRINT_DPI} dpi")
    print()

    # US Letter is 216 x 279 mm; A4 is 210 x 297 mm.
    if total_width_mm > 216 or total_height_mm > 279:
        print("  NOTE: this is larger than US Letter. Print it at a copy shop "
              "at 100% scale, or reduce CHARUCO_SQUARE_LENGTH_MM.")
    else:
        print("  This fits on US Letter. Print at 100% / 'Actual size'.")
    print()
    print("  AFTER PRINTING: measure the squares and update camera_config.py.")


if __name__ == "__main__":
    main()
