import os
import shutil
import json
import cv2

# === Đường dẫn gốc ===
ROOT = r""
RAW_IMG_DIR = os.path.join(ROOT, "rawdata", "images")
RAW_JSON_DIR = os.path.join(ROOT, "rawdata", "labels")
OUT_IMG_DIR = os.path.join(ROOT, "seg_data", "images")
OUT_LABEL_DIR = os.path.join(ROOT, "seg_data", "labels")

# Tạo folder output nếu chưa có
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

CLASS_ID = 0  # class "board"

def convert_and_copy(image_name):
    name, ext = os.path.splitext(image_name)
    if ext.lower() not in [".jpg", ".jpeg", ".png"]:
        return

    img_path = os.path.join(RAW_IMG_DIR, image_name)
    json_path = os.path.join(RAW_JSON_DIR, name + ".json")
    out_img_path = os.path.join(OUT_IMG_DIR, image_name)
    out_label_path = os.path.join(OUT_LABEL_DIR, name + ".txt")

    if not os.path.exists(json_path):
        print(f"⚠️ Không tìm thấy JSON cho {image_name}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_name}")
        return
    h, w = img.shape[:2]

    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data or "content" not in data[0] or len(data[0]["content"]) < 4:
        print(f"⚠️ JSON không hợp lệ: {json_path}")
        return

    coords = []
    for pt in data[0]["content"]:
        x = pt["x"] / w
        y = pt["y"] / h
        coords.extend([x, y])

    # Ghi file label .txt
    with open(out_label_path, "w") as out:
        out.write(f"{CLASS_ID} " + " ".join(f"{c:.6f}" for c in coords) + "\n")

    # Copy ảnh
    shutil.copy(img_path, out_img_path)
    print(f"✅ {image_name} → Đã xử lý và lưu vào processed_data/")

def process_all():
    for fname in os.listdir(RAW_IMG_DIR):
        convert_and_copy(fname)

if __name__ == "__main__":
    process_all()
