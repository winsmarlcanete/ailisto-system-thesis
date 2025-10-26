import torch
from ultralytics import YOLO

# === 1. Define paths ===
DATA_YAML = "student_detection/datasets/SCB5-Handrise-Read-write/SCB5-Handrise-Read-write.yaml"
PROJECT = "student_detection/yolov5_main/runs/train"
WEIGHTS = "yolov5m.pt"
NAME = "yolov5m_SCB5-Handrise-Read-write"

# === 2. Training configuration ===
EPOCHS = 100
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
