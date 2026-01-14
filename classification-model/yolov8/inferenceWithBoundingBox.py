from ultralytics import YOLO
import cv2
import torch

def main():
    video_path = "classification-model/data/videos/Part 1 to 3.mp4"
    model_path = "classification-model/yolov8/trained-weights/yolov8m.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=0.25,
            iou=0.5,
            device=device,
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
