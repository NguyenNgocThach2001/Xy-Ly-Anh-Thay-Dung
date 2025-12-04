from ultralytics import YOLO

def train_yolov8():
    base_model = "yolov8m.pt"    # yolo model
    data_yaml = "Dataset_Detection/piece/Dataset_combined/data.yaml" # duong dan file data.yaml, chua thong tin duong dan dataset (train, validate, test)
    epochs = 500 # 500 epoch
    imgsz = 640 # kich thuoc input
    batch = 16 

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="model", #output
        mosaic=1.0, 
        save_period=50
    )

if __name__ == "__main__":
    train_yolov8()
