# Ai Listo

Ai Listo is a student attention monitoring system designed to support classroom analysis using computer vision and machine learning. The system focuses on detecting visual cues from students and classifying their attention state in real time.

The project uses YOLOv8 to detect students and key body parts such as head, hands, and torso. From these detections, lightweight features like bounding box geometry and motion patterns are extracted instead of using heavy image based models. These features are then passed to an XGBoost classifier to determine student attention levels.

Ai Listo aims to be fast, lightweight, and suitable for real time classroom environments while maintaining good accuracy.

## Key Features
- Real time student detection using YOLOv8
- Multi cue feature extraction from head orientation and posture
- Attention state classification using XGBoost

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- XGBoost
- OpenCV
- Project Status

**This project is currently under development as part of an academic research and thesis work.**
