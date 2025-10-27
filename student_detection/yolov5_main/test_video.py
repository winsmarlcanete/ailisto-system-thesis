from ultralytics import YOLO
import cv2

# Load both trained models
student_model = YOLO("student_detection/yolov5_main/runs/train/yolov5m_student/weights/best.pt")
teacher_model = YOLO("student_detection/yolov5_main/runs/train/yolov5m_teacher/weights/best.pt")

video_path = "student_detection/test_videos/classroom_test.mp4"

cap = cv2.VideoCapture(video_path)

# Video output settings (optional)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(
    "student_detection/yolov5_main/runs/test/combined_live_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
)

# Process video frame-by-frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detections
    student_results = student_model(frame, conf=0.68, verbose=False)
    teacher_results = teacher_model(frame, conf=0.68, verbose=False)

    # Annotate both models correctly
    annotated_frame = frame.copy()
    annotated_frame = student_results[0].plot(img=annotated_frame)
    annotated_frame = teacher_results[0].plot(img=annotated_frame)

    cv2.imshow("Live Teacher + Student Detection", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n✅ Live detection complete!")
print("Saved combined output video at: student_detection/yolov5_main/runs/test/combined_live_output.mp4")
