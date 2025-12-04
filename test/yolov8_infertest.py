from ultralytics import YOLO

def test_model():
    model_path = 'runs/detect/xiangqi3/weights/best.pt'
    data_yaml = 'dataset/piece/data.yaml'

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split='test')

    # TRUY CẬP ĐÚNG CÁC METRIC (không gọi như hàm)
    print("\n📊 Evaluation on TEST set:")
    print(f"Precision (mp)       : {metrics.box.mp:.4f}")
    print(f"Recall    (mr)       : {metrics.box.mr:.4f}")
    print(f"mAP@0.5   (map50)    : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 (map)   : {metrics.box.map:.4f}")

    print("\nRunning prediction and saving images with bounding boxes...")
    model.predict(
        source='dataset/piece/test/images',
        save=True,
        conf=0.25,
        iou=0.45,
        imgsz=640
    )
    print("Dự đoán hoàn tất. Ảnh lưu ở thư mục: runs/predict/")

if __name__ == "__main__":
    test_model()
