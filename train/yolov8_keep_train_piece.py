from ultralytics import YOLO

def train_yolov8_resume():
    # === CẤU HÌNH ===
    pretrained_weights = "runs/detect/model_on_dataset_combined_500_epoch_2/weights/last.pt"  # model đã train trước đó
    data_yaml = "dataset/piece/Dataset_combined/data.yaml"              # file cấu hình dataset
    epochs = 100        # train thêm 50 epochs nữa
    imgsz = 640
    batch = 8

    # Load model đã huấn luyện
    model = YOLO(pretrained_weights)

    # Train tiếp tục
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="model_on_dataset_combined_500_epoch_2",
        resume=True  # Không dùng checkpoint Ultralytics, chỉ load trọng số từ last.pt
    )

    print("✅ Tiếp tục huấn luyện hoàn tất!")

if __name__ == "__main__":
    train_yolov8_resume()
