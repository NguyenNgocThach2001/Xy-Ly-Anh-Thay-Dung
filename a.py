from ultralytics import YOLO

model = YOLO("runs/detect/model_on_dataset_combined_500_epoch_medium/weights/best.pt")
print(model.info())