import importlib
import subprocess
import sys
import torch

def check_ultralytics():
    if importlib.util.find_spec("ultralytics") is None:
        print("Ultralytics package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

    else: print("ultralytics already installed")

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    return device

check_ultralytics()

from ultralytics import YOLO

device = get_device()

model = YOLO("yolov8m.pt") 

model.to(device)

print(f"Model loaded on {device}")


