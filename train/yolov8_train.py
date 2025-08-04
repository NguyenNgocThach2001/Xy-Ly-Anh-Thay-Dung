from ultralytics import YOLO
import os

def train_kfold_yolov8():
    # === CẤU HÌNH ===
    base_model = "yolov8n.pt"  # bạn có thể đổi thành yolov8s.pt hoặc yolov8m.pt nếu cần
    k_folds = 5  # Số fold bạn đã tạo
    config_base_path = "dataset/config_fold{}.yaml"
    epochs = 50
    imgsz = 640
    batch = 16

    # === TRAIN TỪNG FOLD ===
    for fold in range(1, k_folds + 1):
        print(f"\n🚀 Training fold {fold}...")

        config_path = config_base_path.format(fold)
        if not os.path.exists(config_path):
            print(f"❌ Không tìm thấy: {config_path}")
            continue

        # Khởi tạo model từ pretrained
        model = YOLO(base_model)

        # Train model
        model.train(
            data=config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=f"xiangqi_fold{fold}"
        )

        print(f"✅ Đã train xong fold {fold}")


if __name__ == "__main__":
    train_kfold_yolov8()
