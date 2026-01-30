from ultralytics import YOLO

def train_yolov8():
    """
    YOLOv8 training script based on official YOLOv8 documentation
    https://yolov8.com/
    """

    # -------------------------------------------------
    # 1. Load a pretrained YOLOv8 model
    # -------------------------------------------------
    # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO("yolov8n.pt")

    # -------------------------------------------------
    # 2. Train the model on a custom dataset
    # -------------------------------------------------
    model.train(
        data="C:\\Users\\Jerwin\\Documents\\Thesis\\ailisto-system-thesis\\classification-model\\data\\new-dataset\\data.yaml",  # dataset configuration file
        epochs=100,                # number of training epochs
        imgsz=640,                 # image size
        batch=16,                   # batch size (reduced for CPU training)
        device=0,              # 0 = GPU, "cpu" = CPU
        workers=6,                 # dataloader workers (reduced for CPU)
        project="new-dataset-training",        # project folder
        name="yolov8_attention",   # experiment name
        pretrained=True,
        optimizer="AdamW",
        patience=20,
        verbose=True
    )

    # -------------------------------------------------
    # 3. Validate the trained model
    # -------------------------------------------------
    metrics = model.val()
    print("Validation results:", metrics)

    # -------------------------------------------------
    # 4. Export the trained model
    # -------------------------------------------------
    model.export(format="onnx")  # for deployment or further processing

if __name__ == "__main__":
    train_yolov8()
