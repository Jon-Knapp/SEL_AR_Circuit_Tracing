# Augmented Reality Circuit Tracing
 
An overhead computer-vision tool that helps a technician **record and verify wiring
connections** on Schweitzer Engineering Laboratories (SEL) terminal-block panels.
A camera mounted above the board identifies the panel components and tracks the two
test probes, while a LabJack U12 reports whether the probe tips are electrically
connected. A connection is written to the record **only when the vision system and
the hardware sensor agree**. The system is a verification and record-keeping aid
that augments the technician's judgment, not a replacement for it.
 
This project was built as a Portland State University senior capstone for SEL.
 
> **Runs fully offline.** The system makes no network calls and requires no cloud
> services. All dependencies are installed once at setup and run entirely on the
> local machine thereafter.
 
---
 
## What it does
 
1. **Detects panel components** with a YOLOv8 oriented-bounding-box (OBB) model:
   `Flathead_Block`, `Phillips_Block`, `Terminal_1`, and `Terminal_2`.
2. **Stamps a terminal map** onto every detected device. Terminal positions are
   learned once (per device class) during calibration and stored as fractions
   inside each device's bounding box, so the same calibration applies to every
   device of that class, anywhere on the board.
3. **Tracks the two probes** by color (HSV). No physical markers are attached to
   the probes. Instead, you teach each probe its color by clicking it in the live video.
4. **Flattens the board for the record** using four ArUco markers to compute a
   homography. The flattened view is a clean record surface only; nothing is
   tracked on it.
5. **Records connections** automatically. When the LabJack reports continuity *and*
   the camera sees each probe resting on a known, different terminal, and that
   holds steady past a short debounce, the pair is recorded, grouped with anything
   already wired to it, and written to a `.txt` table and a SQLite database.
### Decision-gated recording (the core design principle)
 
The system never records a connection on the strength of the camera alone or the
sensor alone. Both must agree. This is a deliberate safeguard against automation
bias: a recorded pair is only as trustworthy as the terminal IDs underneath it, and
the system is built to make its uncertainty visible rather than to assert a
connection it cannot stand behind. Where it cannot be sure, it says so on screen
(for example, a bold "LABJACK NOT CONNECTED" banner whenever the sensor is offline,
stamped onto saved records as well so an image never silently implies continuity was
being measured).
 
---
 
## Hardware required
 
- **Elgato Facecam 4K**, mounted overhead, roughly perpendicular to the board.
  (Configured via the Elgato Camera Hub to Mirror + Flip; see `config.py`.)
- **LabJack U12**, connected by USB, wired to the **two test probes** and *not* to the
  circuit terminals directly. The U12 reports a single yes/no: are the probe tips
  electrically connected?
- **Two test probes**, with no markers attached.
- **Four ArUco markers** (IDs 0, 3, 4, 5; dictionary `DICT_7X7_50`) at the corners
  of the work surface.
- A Windows PC. A CUDA-capable GPU speeds up detection but is **not required** —
  the detector runs only once per session, so a CPU is fine.
---
 
## Software setup
 
Developed and tested on **Python 3.13, Windows**. Python 3.10 or newer is expected
to work. There are two installation steps: the Python packages, and the LabJack U12
hardware driver.
 
### 1. Python packages
 
```bash
# from the repository root
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
 
### 2. LabJack U12 Windows driver
 
`pip` installs the Python wrapper (the `LabJackPython` package, which is what
provides `import u12`) but **not** the hardware driver. On Windows the U12 needs its
driver installed separately:
 
1. Download and run **`LabJack-U12-Installer-2023-09-05.exe`** from LabJack's
   [U12 software page](https://support.labjack.com/docs/u12-software-installer-downloads-u12).
2. This installs the U12 Library (`ljackuw.dll`, **v1.22**) plus LabJack's utilities
   and documentation. It is compatible with Windows through Windows 11.
3. Verify the sensor on its own before running the full system:
```bash
   cd tools
   python test_labjack_u12.py
```
   Shorting the two probe tips should print `connected`; separating them prints
   `open`.
 
---
 
## One-time calibration
 
The repository ships with a `terminal_map.json` already calibrated for the delivered
panel, so you can run the system as-is. You only need to recalibrate if you change
the camera resolution, the camera position, or the set of device classes.
 
To recalibrate, run the calibration tool from inside `src/` and follow the on-screen
controls:
 
```bash
cd src
python calibrate_terminals.py
```
 
You teach the probe its color, pick a device class, rest the probe on each terminal
and press `x` to mark it, then press `q` to save. The tool writes a new
`terminal_map.json`. **Run it at the same camera resolution you run the main system
at.**
 
---
 
## Running the system
 
```bash
cd src
python main.py
```
 
Run from inside `src/` so the program finds `weights_v2_4_obb.pt`,
`terminal_map.json`, and writes its output folders alongside the code.
 
### On-screen controls
 
| Key        | Action                                              |
|------------|-----------------------------------------------------|
| `1` / `2`  | choose which probe to teach                         |
| click + `s`| teach the chosen probe its color                    |
| `g`        | show / hide terminal circles + connection flags     |
| `c`        | save a clean record image + a labeled legend image  |
| `r`        | start / stop recording a video of the session       |
| `u`        | undo the most recent recorded connection            |
| `o`        | re-run component detection                           |
| `l`        | re-lock the homography (re-find the ArUco markers)  |
| `d`        | toggle the probe color-mask debug windows           |
| `h`        | show / hide the on-screen controls window           |
| `q`        | save the final record and quit                      |
 
---
 
## What gets saved
 
Each session writes to local folders (created automatically, ignored by Git):
 
- **`captures/`**: clean record images and labeled legend images (`c` and `q`).
- **`recordings/`**: session videos (`r`).
- **`connections/`**: a fresh, timestamped pair of files per session: a
  human-readable `.txt` table and a SQLite `.db`. A previous session's files are
  never overwritten.
The list of recorded connections is the single source of truth: the groups, the
`.txt`, and the database are rebuilt from it on every change, so undo is always
consistent across all three.
 
---
 
## Repository structure
 
```
SEL_AR_Circuit_Tracing/
├── src/                  the final, runnable system (run from here)
├── tools/                supporting scripts used to build the project
├── weights/
│   ├── device_detection/ training run behind the delivered detection model
│   └── probe_tracking/    future-work model (NOT used by the delivered system)
├── datasets/
│   ├── device_detection/ dataset that trained the delivered detection model
│   └── probe_detection/   pointer to the future-work dataset (see below)
├── docs/                 final report, supporting documents, and a media pointer
├── requirements.txt
├── LICENSE
└── README.md
```
 
Some large files are hosted as GitHub **Release** assets rather than committed to
the repository, because they exceed GitHub's 100 MB per-file limit:
 
- The **probe-detection dataset** (~476 MB) is an experimental, future-work dataset
  **not used by the delivered system**. See `datasets/probe_detection/README.md`.
- Three **sponsor-requested videos** (~1 GB each). See `docs/media.md`.
Both are available from the [v1.0 release](https://github.com/Jon-Knapp/SEL_AR_Circuit_Tracing/releases/tag/v1.0).
 
---
 
## Known limitations
 
These are documented honestly so the system is not trusted beyond what it can do:
 
- **Terminal-ID trust.** A recorded pair is only as trustworthy as the terminal IDs
  the vision system assigned. The debounce rejects a momentary brush but cannot catch
  a probe resting steadily on a mis-identified terminal. The on-screen circles and
  the saved legend image make every assignment visible and traceable.
- **180° flip ambiguity.** A symmetric rectangular device looks the same rotated
  end-for-end, so if a device is placed reversed from how it was calibrated, its
  terminal numbering flips. Devices placed the usual way up are unaffected. Resolving
  this is future work.
- **LabJack reconnect.** Recovery after the U12 is unplugged and replugged
  mid-session is not guaranteed; restarting the program is the clean recovery path.
- **Cross-session queries.** Each session records to its own files; querying across
  multiple sessions requires opening more than one database.
---
 
## Future work
 
- Markerless probe tracking with a trained object detector (dataset attached to the
  v1.0 Release).
- A natural-language query layer over the recorded connections using a locally hosted
  LLM (preserving the offline constraint).
- Improved probe-tracking robustness and a resolution to the 180° flip ambiguity.
---
 
## License
 
See [`LICENSE`](LICENSE).
 
Built for Schweitzer Engineering Laboratories.
