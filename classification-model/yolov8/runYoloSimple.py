from ultralytics import YOLO
import torch

def main():
    video_path = "classification-model/data/videos/Part 1 to 3.mp4"
    model_path = "classification-model/yolov8/trained-weights/yolov8m.pt" 
    output_dir = "output/runs/simple"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = YOLO(model_path)
    model.to(device)

    results = model.predict(
        source=video_path,
        save=True,          # saves output video with boxes
        save_txt=False,     
        conf=0.25,          
        iou=0.5,            
        device=device,
        project=output_dir,
        name="simple_test"  
    )

    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
