from ultralytics import YOLO

# build a new model from scratch
model = YOLO("yolov8n.yaml")

#lets train the model
model.train(data="data.yaml", epochs=1) 