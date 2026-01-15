import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==== CHANGE PATH TO YOUR MODEL FILE ====
model_path = "classification-model/posture-module/trained-weight/pose_landmarker_lite.task"

# ==== SETUP MEDIAPIPE POSE LANDMARKER ====
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)

# ==== OPEN VIDEO / WEBCAM ====
video_source = "classification-model/data/videos/Part 1 to 3.mp4"  # 0 for webcam, or path to video file
cap = cv2.VideoCapture(video_source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_idx = 0

with PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or cannot read frame.")
            break

        # ==== CONVERT FRAME TO RGB ====
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ==== WRAP IN mediapipe.Image ====
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ==== CALCULATE TIMESTAMP (ms) ====
        frame_timestamp_ms = int(frame_idx * (1000.0 / fps))
        frame_idx += 1

        # ==== RUN INFERENCE ====
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # ==== HANDLE & DRAW RESULTS ====
        if result and result.pose_landmarks:
            for pose in result.pose_landmarks:
                for i, lm in enumerate(pose):
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # ==== SHOW OUTPUT ====
        cv2.imshow("Pose Landmarker Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
