import os
import csv
import numpy as np
from ultralytics import YOLO
import torch

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "classification-model\videos\Part 1 to 3.mp4"
MODEL_PATH = "yolov8s.pt"
OUTPUT_DIR = "output/runs/simple2"
CONF_THRES = 0.25
IOU_THRES = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "detections.csv")
NPY_PATH = os.path.join(OUTPUT_DIR, "detections.npy")

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)
model.to(device)

# -----------------------------
# RUN TRACKING
# -----------------------------
results = model.track(
    source=VIDEO_PATH,
    conf=CONF_THRES,
    iou=IOU_THRES,
    persist=True,
    device=device,
    stream=True
)

all_rows = []

# -----------------------------
# PROCESS PER FRAME
# -----------------------------
for frame_idx, r in enumerate(results):
    if r.boxes is None:
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    ids = (
        r.boxes.id.cpu().numpy().astype(int)
        if r.boxes.id is not None
        else [-1] * len(boxes)
    )

    for box, score, cls, track_id in zip(boxes, scores, classes, ids):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # -----------------------------
        # CLASS INTERPRETATION
        # -----------------------------
        label = model.names[cls]

        cue_type = None
        torso_box = None

        if label == "person":
            cue_type = "torso"
            torso_box = [
                x1,
                y1 + height * 0.25,
                x2,
                y1 + height * 0.75
            ]

        elif label in ["face", "head"]:
            cue_type = "head"

        elif label in ["hand", "left hand", "right hand"]:
            cue_type = "hand"

        else:
            continue

        if cue_type == "torso":
            bx1, by1, bx2, by2 = torso_box
        else:
            bx1, by1, bx2, by2 = x1, y1, x2, y2

        all_rows.append([
            frame_idx,
            int(track_id),
            cue_type,
            float(bx1),
            float(by1),
            float(bx2),
            float(by2),
            float(score)
        ])

# -----------------------------
# SAVE CSV
# -----------------------------
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame_index",
        "student_id",
        "cue_type",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence"
    ])
    writer.writerows(all_rows)

# -----------------------------
# SAVE NUMPY
# -----------------------------
np.save(NPY_PATH, np.array(all_rows, dtype=object))

print("Extraction finished")
print(f"CSV saved to: {CSV_PATH}")
print(f"NumPy saved to: {NPY_PATH}")
