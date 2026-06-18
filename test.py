from ultralytics import YOLO

# 1. Load the YOLO26 Nano model (it auto-downloads on the first run)
model = YOLO("yolo26n.pt")

# 2. Run object detection on a local image or a web URL
results = model("https://ultralytics.com/images/bus.jpg")

# 3. Display or save results to the local 'runs/detect/predict' folder
for result in results:
    result.show()  # Opens a window showing the boxes
    result.save()  # Saves the image with bounding boxes to disk
