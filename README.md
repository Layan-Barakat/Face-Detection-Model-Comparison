# Face Detection Model Comparison

Compares **Haar Cascade**, **OpenCV DNN (Caffe)**, and **YOLOv8** face detectors on a single image, measuring detections and processing time, and showing a side-by-side visual comparison.

---

## What it does
- Loads one input image
- Runs Haar, DNN, and YOLOv8
- Times each method and counts faces
- Shows a 3-panel result (Haar = Green, DNN = Blue, YOLO = Red)

---

## Structure
```text
.
├─ scripts/
│  └─ face_comparison.py
├─ models/
│  ├─ deploy.prototxt
│  └─ res10_300x300_ssd_iter_140000.caffemodel
├─ requirements.txt
└─ README.md
```

> Put the **Caffe model files** in `models/`:
> - `deploy.prototxt`
> - `res10_300x300_ssd_iter_140000.caffemodel`

---

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run
```bash
python scripts/face_comparison.py --image path/to/your/image.jpg
```

A window will pop up with three columns:
- **Haar (Green)** | **DNN (Blue)** | **YOLO (Red)**  

The terminal prints a summary table with counts + timing.

---

## Notes
- Haar: fastest but least robust  
- DNN: balanced speed/accuracy  
- YOLOv8: best accuracy but slower  

---

## Author
Layan Barakat — University of Birmingham Dubai
