# Car Detector / Counter (YOLOv8 + SORT)

Python scripts for:
- Running YOLO detection on an image.
- Running YOLO detection on webcam / video.
- Counting cars in a video using YOLO + SORT tracking.

This repo is currently Windows-first and several scripts contain **hard-coded absolute paths** (e.g. `d:/...`). You will likely need to edit those paths to match your machine.

## Project structure

- `car counter/`
  - `Car-Counter.py` — detects cars (no tracking/counting line).
  - `CountingCars.py` — car counting using SORT + a counting line.
  - `sortFilter.py` — SORT tracker implementation.
  - `test.py` — alternative tracking/counting test with FPS display.
- `yolo running/`
  - `yolo_run.py` — YOLO on a single image.
- `yolo with webcam and videos/`
  - `yolo_webcam.py` — YOLO on webcam or a recorded video.
- `yolo weight/`
  - Put your YOLO model weights here (not committed to git).

## Requirements

- Python 3.9+ (recommended)
- Packages:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `filterpy` (used by SORT)
  - `scipy` (fallback for assignment in SORT)
  - `matplotlib`, `scikit-image` (imported by `sortFilter.py`)

Install (from the repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy filterpy scipy matplotlib scikit-image
```

Notes:
- `sortFilter.py` tries to use `lap` for faster assignment; if you want it, install it separately (optional).

## Model weights (required)

Weights are **not included** in the repo (and are ignored by `.gitignore`) because they are large.

Your scripts reference these files:
- `yolo weight/yolov8n.pt`
- `yolo weight/yolov8l.pt`
- `yolo weight/yolo11n.pt` (used in `yolo running/yolo_run.py`)

You can either:
1) Download the weights manually and place them into `yolo weight/`, OR
2) Edit the scripts to load a model name that Ultralytics auto-downloads (for example `YOLO('yolov8n.pt')`).

If you keep the current paths, make sure the files exist here:

```
yolo weight/
  yolov8n.pt
  yolov8l.pt
  yolo11n.pt
```

## Media files (images/videos)

Sample videos/images are also ignored by `.gitignore` (they were too large for GitHub).

To run the scripts, put your own files in:
- `yolo running/Images/`
- `yolo running/Videos/`

…and update the paths inside the scripts if needed.

## Run

Because some folders have spaces, use quotes in PowerShell.

### 1) Detect objects in an image

```powershell
python "yolo running/yolo_run.py"
```

You may need to edit `yolo running/yolo_run.py` to point to an existing image on your machine.

### 2) Detect objects in a webcam/video stream

```powershell
python "yolo with webcam and videos/yolo_webcam.py"
```

Inside the file there are 2 capture options:
- Webcam: `cv2.VideoCapture(0)`
- Video file: `cv2.VideoCapture(r'd:/.../bicycle.mp4')`

Comment/uncomment and fix the video path as needed.

### 3) Car detection (no counting)

```powershell
python "car counter/Car-Counter.py"
```

### 4) Car counting (YOLO + SORT)

```powershell
python "car counter/CountingCars.py"
```

This script:
- Detects cars with YOLO.
- Tracks objects with SORT.
- Counts a car when its tracked center crosses `limit_line`.

Adjust these settings in the file for your video:
- The input video path (`cv2.VideoCapture(...)`).
- `limit_line` coordinates.
- Confidence/area/aspect ratio thresholds.

### Exit

Press `q` in the OpenCV window to quit.

## Troubleshooting

- **Black window / crash**: usually a wrong video/image path. Fix the `cv2.VideoCapture(...)` path.
- **No detections**: check the weights path and that the `.pt` file exists.
- **Import errors from `sortFilter.py`**: install missing deps (`filterpy`, `scipy`, `matplotlib`, `scikit-image`).

## License note

`car counter/sortFilter.py` contains a GPL license header from the SORT project. If you plan to redistribute this repository, review that file and ensure your distribution complies with its license terms.
