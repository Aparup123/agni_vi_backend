from ultralytics import YOLO
import shutil
import os

# Delete the 'runs' directory
runs_dir = 'runs'
if os.path.exists(runs_dir):
    shutil.rmtree(runs_dir)

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Run inference on 'crater1.jpg' with specified arguments
results = model.predict("crater1.jpg", save=True, imgsz=320, conf=0.25, show_labels=False)

# Define paths
source_file = 'runs/detect/predict/crater1.jpg'
destination_dir = 'output/'
new_filename = 'final.jpg'  # Rename the file to 'new_name.jpg'
destination_file = os.path.join(destination_dir, new_filename)

# Ensure the output directory exists
os.makedirs(destination_dir, exist_ok=True)

# Copy the file
if os.path.exists(source_file):
    shutil.copy2(source_file, destination_file)