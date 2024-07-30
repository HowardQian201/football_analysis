
from ultralytics import YOLO

# model = YOLO("yolov8l") # change model to yolov8x  yolov8l yolov8m yolov8s if computer cant run
model = YOLO("models/best.pt")


results = model.predict("input_videos/08fd33_4.mp4", save=True)
print(results[0])
print("---------")
for box in results[0].boxes:
    print(box)



