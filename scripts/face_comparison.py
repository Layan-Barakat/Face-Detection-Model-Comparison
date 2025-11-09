import argparse
import time
import numpy as np
import cv2
from ultralytics import YOLO


def draw_boxes(img, boxes, color, label=None):
    """Draw rectangles for detected faces."""
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(out, label, (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def detect_haar(img):
    """Detect faces using Haar Cascade."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades +
                                 "haarcascade_frontalface_default.xml")
    t0 = time.time()
    faces = haar.detectMultiScale(gray, scaleFactor=1.1,
                                  minNeighbors=5, minSize=(30, 30))
    t1 = time.time()
    boxes = [(x, y, w, h) for (x, y, w, h) in faces]
    return boxes, t1 - t0


def detect_dnn(img):
    """Detect faces using OpenCV DNN (Caffe model)."""
    h, w = img.shape[:2]
    net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt",
                                   "models/res10_300x300_ssd_iter_140000.caffemodel")
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    t0 = time.time()
    net.setInput(blob)
    det = net.forward()
    t1 = time.time()

    boxes = []
    for i in range(det.shape[2]):
        conf = det[0, 0, i, 2]
        if conf > 0.6:
            x1, y1, x2, y2 = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes, t1 - t0


def detect_yolo(img):
    """Detect faces using YOLOv8n."""
    model = YOLO("yolov8n.pt")  # downloads automatically if missing
    t0 = time.time()
    results = model.predict(img, stream=False, verbose=False)
    t1 = time.time()

    boxes = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes, t1 - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print("Image not found:", args.image)
        return

    haar_boxes, t_haar = detect_haar(frame)
    dnn_boxes, t_dnn = detect_dnn(frame)
    yolo_boxes, t_yolo = detect_yolo(frame)

    print("Method | Faces | Time (s)")
    print(f"Haar   | {len(haar_boxes):5d} | {t_haar:.3f}")
    print(f"DNN    | {len(dnn_boxes):5d} | {t_dnn:.3f}")
    print(f"YOLO   | {len(yolo_boxes):5d} | {t_yolo:.3f}")

    img_h_
