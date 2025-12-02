from ultralytics import YOLO

def continue_training_after_finish():
    # 1) load từ best/last của run cũ
    weights = "runs/detect/model_on_dataset_combined_500_epoch_small/weights/best.pt"
    model = YOLO(weights)

    # 2) train TIẾP như run mới (không resume)
    results = model.train(
        data="dataset/piece/Dataset_combined/data.yaml",  # hoặc YAML đã gộp
        epochs=500,       # số epoch bạn muốn thêm
        imgsz=640,
        batch=16,
        lr0=1e-3,         # giảm LR để fine-tune ổn định
        mosaic=0.3,       # fine-tune nên thấp (0.0–0.3)
        freeze=0,         # nếu dữ liệu mới rất ít, thử freeze=10
        name="500_epoch_2_small_continue_mixed_500_more_epoch",   # TÊN MỚI, tránh đè run cũ
        project="runs/detect",
        save_period=50,
    )

if __name__ == "__main__":
    continue_training_after_finish()
