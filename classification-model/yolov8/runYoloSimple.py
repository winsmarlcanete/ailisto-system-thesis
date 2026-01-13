from ultralytics import YOLO
import torch

def main():
    video_path = r"C:\Users\Jerwin\Documents\Thesis\ailisto-system-thesis\classification-model\videos\Part 1 to 3.mp4"  # Palitan mo to jer
    model_path = "yolov8m.pt" 
    output_dir = "output/runs/simple"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = YOLO(model_path)
    model.to(device)

    results = model.predict(
        source=video_path,
        save=True,          # saves output video with boxes
        save_txt=False,     # set True if you want txt labels
        conf=0.25,          # confidence threshold
        iou=0.5,            # IoU threshold
        device=device,
        project=output_dir,
        name="simple_test"  #kaya simple kasi wala pang cue modules, tas yung runYoloSimple2.py baseline yolov8 din, itrtry idetect each cue module ng wlang embedded cue module na pangdetect separately gets mo ba jermanno
    )

    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
