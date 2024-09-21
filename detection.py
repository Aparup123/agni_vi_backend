from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("best.pt")

# Run inference on 'crater1.jpg' with specified arguments
results = model.predict("crater1.jpg", save=True, imgsz=320, conf=0.20, show_labels=False)
