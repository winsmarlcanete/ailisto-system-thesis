import torch
from ultralytics import YOLO


DATA_YAML = "student_detection/datasets/scb_teacher/scb_teacher_data.yaml"  
PROJECT = "student_detection/yolov5_main/runs/train"
WEIGHTS = "yolov5m.pt"   
NAME = "yolov5m_teacher"


EPOCHS = 50              
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    print("🚀 Starting YOLOv5 teacher training...")
    model = YOLO(WEIGHTS)

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
