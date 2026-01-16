import cv2
import mediapipe as mp
from ultralytics import YOLO
import os

VIDEO_PATH = os.path.abspath("C:/Users/Windows/Desktop/VS Code/ailisto-main/ailisto-system-thesis/classification-model/data/videos/sample_vid.mp4")
YOLO_MODEL_PATH = "yolov8n.pt"
POSE_MODEL_PATH = os.path.abspath(
    "C:/Users/Windows/Desktop/VS Code/ailisto-main/ailisto-system-thesis/classification-model/posture-module/trained-weight/pose_landmarker_lite.task"
)


print("Pose model path:", POSE_MODEL_PATH)

# Test if file exists
if not os.path.exists(POSE_MODEL_PATH):
    raise FileNotFoundError(f"PoseLandmarker task file not found at {POSE_MODEL_PATH}")

PERSON_CLASS_ID = 0
CROP_PADDING = 20
MIN_CROP_SIZE = 80

# Load models once
yolo = YOLO(YOLO_MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video!")
    else:
        print("Video opened successfully")

    frame_index = 0

    with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame")
                break

            frame_index += 1
            if frame_index % 30 == 0:
                print(f"Processing frame {frame_index}")

            # Encode as JPEG
            try:
                _, buffer = cv2.imencode(".jpg", frame)
            except Exception as e:
                print("ERROR encoding frame:", e)
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

