# Camera Calibration and Pose Measurement

Two measurement tools for the Continuity Annotation System, supporting the
Camera Position Evaluation Plan.

**Nothing here changes how `main.py` behaves.** These are separate tools with
their own config file. They read the same camera and the same four ArUco
markers, and they write their own output files.

---

## What these tools answer

**`calibrate_camera.py`** — what are this camera's optics? Focal length, where
the lens centre actually falls on the sensor, and how much the lens bends
straight lines. These do not change when you move the camera.

**`measure_camera_pose.py`** — where is the camera right now, and which way is
it pointing, relative to the plywood? Reported as a 3D position in millimetres
plus tilt, azimuth, roll, and the aviation yaw/pitch/roll.

It also predicts **probe parallax** at any point on the board, which is the
number that decides whether a camera position is usable at all.

You must do the calibration first. Until the lens distortion is known, an angle
measured from the markers is contaminated — and worst at the frame edges, which
is exactly where the markers sit.

---

## Order of operations

```
1.  python test_pose_math.py               # no hardware needed; proves the maths
2.  python make_charuco_board.py           # generates the printable board
    ... print it, glue it to something rigid, MEASURE the squares ...
    ... put the measured sizes into camera_config.py ...
3.  python capture_calibration_images.py   # photograph the board, ~25 views
4.  python calibrate_camera.py             # writes camera_intrinsics.json
    ... measure the four board markers, fill in MARKER_LAYOUT ...
5.  python measure_camera_pose.py --solve-rotations   # if unsure of rotations
6.  python measure_camera_pose.py          # measure the pose
```

Run step 1 before anything else. If it fails, the problem is in the code. If it
passes and real measurements still look wrong, the problem is in the numbers you
measured — and knowing which saves hours.

---

## Files

| File | What it is |
|---|---|
| `camera_config.py` | Every setting. The only file you normally edit. |
| `make_charuco_board.py` | Generates `charuco_board.png` for printing. |
| `capture_calibration_images.py` | Interactive capture with sharpness and frame-coverage checks. |
| `calibrate_camera.py` | Runs the calibration; writes `camera_intrinsics.json`. Also provides `load_intrinsics()`. |
| `camera_pose.py` | Library. Pose solving, angle decomposition, parallax prediction. No side effects on import. |
| `measure_camera_pose.py` | The runnable pose tool (live, single image, or rotation search). |
| `test_pose_math.py` | Self-check. Needs no hardware. |

---

## The four things most likely to go wrong

**1. You typed the intended print size instead of the measured one.**
Printers scale by a few percent by default. Every millimetre the pose tool
reports is scaled by `CHARUCO_SQUARE_LENGTH_MM`. Measure five squares with a
steel rule and divide by five. Nothing downstream can detect this error.

**2. The calibration resolution does not match the operating resolution.**
Intrinsics are expressed in pixels, so they are only valid at one image size.
Calibrate at 3840 × 2160 if that is what `main.py` runs at. Both tools warn if
they notice a mismatch, but do not rely on the warning.

**3. Camera Hub settings changed between calibration and use.**
Digital zoom, PTZ crop, or aspect correction all change the effective optics.
Lock every Camera Hub setting before you calibrate and do not touch them
afterwards. If `fx / fy` comes out far from 1.000, that is usually the cause.

**4. `MARKER_LAYOUT` is wrong.**
If the reprojection error is more than a few pixels, the marker positions or
rotations are wrong. Run `--solve-rotations` to fix the rotations. If that says
its answer is not decisive, the centre positions are wrong and no rotation can
rescue them — go and re-measure the plywood.

---

## Reading the output

**Tilt, azimuth, and board roll** are the recommended angle set. Tilt is θ and
azimuth is φ from the Camera Position Evaluation Plan.

**Azimuth is reported as undefined when the camera is directly overhead.** That
is correct, not a failure: there is no direction to the camera when it is
straight up, the same way there is no compass bearing to the North Pole once
you are standing on it.

**Yaw, pitch and roll** are also reported, because they were asked for, but they
are mathematically degenerate at the overhead baseline. At 90° of pitch, yaw and
roll become the same rotation and cannot be separated — gimbal lock. Near that
pose those two numbers jitter between frames even with the camera bolted down.
The tool prints a warning when this applies. **Quote tilt, azimuth and board
roll in the report; treat yaw/pitch/roll as supplementary.**

**Board roll in image** is worth watching for a reason unrelated to geometry:
the YOLO model was trained with the board at one particular rotation. If this
number drifts far from its training value, expect detection to degrade — and
that is a dataset problem, not a camera-position problem. Recording it lets you
tell the two apart.

**Frame-to-frame spread** is printed after every `m` measurement. That is the
repeatability of the measurement, and it is the number to quote when you write
"the camera was at 30 degrees." A spread above about 0.2° means something is
moving: check the mount, the lighting, and the markers.

---

## The parallax table

Press `p` after a measurement. For each sample point on the board it predicts
how far the probe's tracked colour will *appear* from where the tip actually is.

Read the largest value against **half your tightest terminal pitch**. Above
that, the system will confidently name the wrong terminal.

Two things this table will show you that the simple `h × tan(θ)` formula cannot:

- The error is **different at different places on the board**, even with the
  camera mounted perfectly overhead. A terminal near the frame edge is viewed
  at an angle regardless of how the camera body is mounted.
- The error has a **direction**, not just a magnitude. It always points away
  from the camera. That means it is systematic and, in principle, correctable —
  unlike random noise.

Set `PROBE_COLOUR_HEIGHT_MM` from an actual measurement of your probe in a
normal working grip. Moving the magenta tape closer to the tip reduces this
height, and it is the cheapest parallax mitigation available to you.

---

## OpenCV version

Written for the modern ArUco API (`cv2.aruco.CharucoDetector`,
`cv2.aruco.ArucoDetector`, `board.matchImagePoints`), which needs **OpenCV
4.7 or later**. Verified on 4.13. The older `interpolateCornersCharuco` and
`calibrateCameraCharuco` helpers are deliberately not used — they are removed or
deprecated in current versions.
