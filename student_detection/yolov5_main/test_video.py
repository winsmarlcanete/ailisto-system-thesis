from ultralytics import YOLO

# Load  trained model 
model = YOLO("student_detection/yolov5_main/runs/train/scb/best.pt")


video_path = "student_detection/test_videos/classroom_test.mp4"


results = model.predict(
    source=video_path,   # video file
    conf=0.68,            # confidence threshold (0–1)
    show=True,           # show real-time window
    save=True,           # save output video
    project="student_detection/yolov5_main/runs/test",  # output folder
    name="video_test_results",  # folder name for results
)

print("\n✅ Video testing complete! Check saved video in:")
print("student_detection/yolov5_main/runs/test/video_test_results")
