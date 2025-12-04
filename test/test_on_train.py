from ultralytics import YOLO
import cv2
import os

def val_and_save_images():
    # === CẤU HÌNH ===
    model_path = "runs/detect/500_epoch_2_small_continue_mixed_500_more_epoch/weights/best.pt"
    data_yaml = "dataset/piece/Dataset5_64image_nghieng/data.yaml"
    imgsz = 640
    batch = 16
    save_dir = "val_results_images"
    os.makedirs(save_dir, exist_ok=True)

    # Load model đã train
    model = YOLO(model_path)

    # Chạy val trên tập train (hoặc đổi split="val" nếu muốn)
    results = model.val(
        data=data_yaml,
        split="train",  # tập train, đổi thành 'val' nếu cần
        imgsz=imgsz,
        batch=batch,
        save=True  # Lưu ảnh dự đoán
    )

    # Lấy tên class
    names = model.names

    # Vẽ lại kết quả trên ảnh với tên class
    for i, result in enumerate(results):
        img = result.orig_img.copy()
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Vẽ bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_path = os.path.join(save_dir, f"result_{i}.jpg")
        cv2.imwrite(out_path, img)

    print(f"✅ Đã lưu ảnh kết quả vào: {save_dir}")

if __name__ == "__main__":
    val_and_save_images()
