import os
import shutil

# === Paths ===
DATASET_DIR = "student_detection/datasets/scb_teacher"
TARGET_CLASS = 4  # 'teacher' index in original dataset
OUT_DIR = "student_detection/datasets/scb_teacher_new"

os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)

def extract_class(split):
    label_dir = f"{DATASET_DIR}/labels/{split}"
    image_dir = f"{DATASET_DIR}/images/{split}"

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))

        with open(label_path, "r") as f:
            lines = f.readlines()

        teacher_lines = []
        for l in lines:
            parts = l.strip().split()
            if parts and int(parts[0]) == TARGET_CLASS:
                # Change class index from 4 → 0 (for clean new dataset)
                parts[0] = "0"
                teacher_lines.append(" ".join(parts) + "\n")

        if teacher_lines:
            out_label_path = f"{OUT_DIR}/labels/{split}/{label_file}"
            with open(out_label_path, "w") as f:
                f.writelines(teacher_lines)

            shutil.copy(image_path, f"{OUT_DIR}/images/{split}/")

for split in ["train", "val"]:
    extract_class(split)

print("✅ Extracted teacher images and labels to:", OUT_DIR)
