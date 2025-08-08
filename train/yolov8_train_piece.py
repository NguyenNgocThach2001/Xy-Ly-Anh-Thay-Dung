from ultralytics import YOLO

def train_yolov8():
    # === CẤU HÌNH ===
    base_model = "yolov8s.pt"
    data_yaml = "dataset/piece/Dataset_combined/data.yaml"
    epochs = 500
    imgsz = 640
    batch = 16

    # Khởi tạo model từ pretrained
    model = YOLO(base_model)

    # Train model kèm mosaic augmentation
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="model_on_dataset_combined_500_epoch_2",
        mosaic=1.0, 
        save_period=50
    )

    print("✅ Huấn luyện hoàn tất!")

if __name__ == "__main__":
    train_yolov8()
