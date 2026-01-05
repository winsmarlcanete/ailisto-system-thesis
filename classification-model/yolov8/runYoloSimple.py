from ultralytics import YOLO
import torch

def main():
    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    video_path = "data/videos/classroom_sample.mp4"  # change this anytime
    model_path = "yolov8s.pt"  # baseline pretrained model
    output_dir = "runs/detect"

    # -----------------------------
    # DEVICE CHECK
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model = YOLO(model_path)
    model.to(device)

    # -----------------------------
    # RUN INFERENCE
    # -----------------------------
    results = model.predict(
        source=video_path,
        save=True,          # saves output video with boxes
        save_txt=False,     # set True if you want txt labels
        conf=0.25,          # confidence threshold
        iou=0.5,            # IoU threshold
        device=device,
        project=output_dir,
        name="baseline"
    )

    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
