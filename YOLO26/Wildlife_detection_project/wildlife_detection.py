!pip install -q -U ultralytics

import os
import yaml
import torch
from ultralytics import YOLO

# Check GPU
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Dataset path
base = "/kaggle/input/datasets/ankanghosh651/object-detection-wildlife-dataset-yolo-format/final_data"

# Check dataset
for folder in [
    "train/images",
    "train/labels",
    "valid/images",
    "valid/labels",
    "test/images",
    "test/labels"
]:
    path = os.path.join(base, folder)
    print(folder, ":", len(os.listdir(path)))

# Load original YAML
original_yaml = os.path.join(base, "data_wl.yaml")

with open(original_yaml, "r") as f:
    data = yaml.safe_load(f)

# Update paths for Kaggle
data["train"] = os.path.join(base, "train/images")
data["val"] = os.path.join(base, "valid/images")
data["test"] = os.path.join(base, "test/images")

data["names"] = {
    0: "buffalo",
    1: "elephant",
    2: "rhino",
    3: "zebra"
}

# Save corrected YAML
yaml_path = "/kaggle/working/wildlife.yaml"

with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("YAML created:", yaml_path)

# Load YOLO26
model = YOLO("yolo26n.pt")

# Train
results = model.train(
    data=yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,
    device=0
)

# Validate
metrics = model.val(
    data=yaml_path,
    split="val",
    device=0
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

# Test images
test_dir = os.path.join(base, "test/images")

images = [
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Test images:", len(images))

# Run prediction
results = model.predict(
    source=images[:10],
    conf=0.25,
    device=0,
    save=True
)

# Show one prediction
results[0].show()
