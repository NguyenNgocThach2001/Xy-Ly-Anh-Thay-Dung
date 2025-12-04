from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ROOT = thư mục Dataset_Segmentation của bạn
ROOT = r"D:\Mon Hoc\Thi Giac May Tinh Thay Duy\Dataset_Segmentation"  # sửa lại cho đúng


def show_seg_sample(root, split="train", index=0):
    root = Path(root)
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"

    # lấy list ảnh
    img_paths = sorted(list(img_dir.glob("*.*")))
    if not img_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {img_dir}")

    # chọn 1 ảnh theo index
    img_path = img_paths[index % len(img_paths)]
    label_path = lbl_dir / (img_path.stem + ".txt")

    print(f"Image : {img_path}")
    print(f"Label : {label_path}")

    # đọc ảnh
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
    h, w = img.shape[:2]

    overlay = img.copy()

    # đọc label và vẽ polygon
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 3:
                    continue  # line không đủ điểm polygon

                cls_id = int(parts[0])
                coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)

                # chuyển từ tọa độ normalized (0–1) -> pixel
                pts = np.zeros_like(coords)
                pts[:, 0] = coords[:, 0] * w  # x * width
                pts[:, 1] = coords[:, 1] * h  # y * height
                pts = pts.astype(np.int32)

                # vẽ mask (fill) + viền polygon
                cv2.fillPoly(overlay, [pts], (0, 255, 0))           # mask xanh lá
                cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)  # viền đỏ

                # (tuỳ chọn) in class_id cho debug
                print(f"  class {cls_id}, {len(pts)} points")

        # trộn overlay với ảnh gốc cho trong suốt
        alpha = 0.4
        vis = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    else:
        print("⚠ Không có file label, chỉ hiển thị ảnh gốc.")
        vis = img

    # hiển thị bằng matplotlib (chuyển BGR -> RGB)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(7, 7))
    plt.imshow(vis_rgb)
    plt.axis("off")
    plt.title(f"{split} - {img_path.name}")
    plt.show()


if __name__ == "__main__":
    # ví dụ: xem ảnh thứ 0 trong train
    show_seg_sample(ROOT, split="train", index=0)
    
    show_seg_sample(ROOT, split="train", index=1)
    
    show_seg_sample(ROOT, split="train", index=4)
    
    show_seg_sample(ROOT, split="train", index=5)
    # thử ảnh khác thì đổi index: 1, 2, 3,...
