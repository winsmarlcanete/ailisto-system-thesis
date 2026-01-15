import cv2
import time
import mediapipe as mp
from ultralytics import YOLO

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===================== CONFIG =====================
VIDEO_PATH = "classification-model/data/videos/Part 1 to 3.mp4"
YOLO_MODEL_PATH = "yolov8n.pt"
POSE_MODEL_PATH = "classification-model/posture-module/trained-weight/pose_landmarker_lite.task"

PERSON_CLASS_ID = 0
CROP_PADDING = 20
MIN_CROP_SIZE = 80
# =================================================


# ===================== YOLO ======================
yolo = YOLO(YOLO_MODEL_PATH)

# ===================== MEDIAPIPE =================
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)


# ===================== VIDEO =====================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_index = 0

with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # ========== YOLO TRACKING ==========
        yolo_results = yolo.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            verbose=False
        )[0]

        if yolo_results.boxes.id is not None:
            boxes = yolo_results.boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if cls_id != PERSON_CLASS_ID:
                    continue

                track_id = int(boxes.id[i])

                x1, y1, x2, y2 = boxes.xyxy[i]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # padding
                x1 -= CROP_PADDING
                y1 -= CROP_PADDING
                x2 += CROP_PADDING
                y2 += CROP_PADDING

                # clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                    continue

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                # ========== MEDIAPIPE POSE ==========
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_crop
                )

                timestamp_ms = int(frame_index * (1000 / fps))

                pose_result = pose_landmarker.detect(mp_image)

                # ========== DRAW POSE ==========
                if pose_result.pose_landmarks:
                    for lm in pose_result.pose_landmarks[0]:
                        px = int(lm.x * (x2 - x1)) + x1
                        py = int(lm.y * (y2 - y1)) + y1
                        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

                # draw bbox + ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        frame_index += 1

        cv2.imshow("YOLO + MediaPipe Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
