import torch
from ultralytics import YOLO

# === 1. Define paths ===
DATA_YAML = "student_detection/datasets/student/data.yaml"
PROJECT = "student_detection/yolov5_main/runs/train"
WEIGHTS = "student_detection/yolov5_main/runs/train/yolov5m_student/weights/best.pt"
NAME = "yolov5m_student"

# === 2. Training configuration ===
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# === 3. Train the model ===
if __name__ == "__main__":
    model = YOLO(WEIGHTS)  # load pretrained model
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=NAME,
        device=DEVICE,
        exist_ok=True
    )

    print(f"\n✅ Training complete! Check results in: {PROJECT}/{NAME}")
