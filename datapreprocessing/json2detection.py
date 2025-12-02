import os
import shutil
import json
import cv2
import yaml

# ========== Cấu hình ==========
ROOT = r""  # ← Đặt đường dẫn gốc (thư mục chứa rawdata/ và processed_data_detection/)
DATA_YAML = os.path.join(ROOT, "Dataset/piece/Dataset_combined/data.yaml")  # ← Trỏ tới data.yaml của bạn

RAW_IMG_DIR = os.path.join(ROOT, "rawdata", "images")
RAW_JSON_DIR = os.path.join(ROOT, "rawdata", "labels")
OUT_IMG_DIR = os.path.join(ROOT, "processed_data_detection", "images")
OUT_LABEL_DIR = os.path.join(ROOT, "processed_data_detection", "labels")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# ========== Đọc class map từ data.yaml ==========
with open(DATA_YAML, "r", encoding="utf-8") as f:
    y = yaml.safe_load(f)

names = y.get("names", [])
name_to_id = {n: i for i, n in enumerate(names)}

# Nếu muốn chuẩn hoá tên (ví dụ tool cũ dùng "guard")
NAME_NORMALIZERS = {
    "guard": "advisor",
}

def normalize_label(name: str) -> str:
    n = name.strip()
    for src, dst in NAME_NORMALIZERS.items():
        n = n.replace(src, dst)
    return n

# ========== Helpers ==========
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def polygon_to_bbox(polygon):
    xs = [pt["x"] for pt in polygon]
    ys = [pt["y"] for pt in polygon]
    return min(xs), min(ys), max(xs), max(ys)

def rectmask_to_bbox(rect):
    x_min = float(rect["xMin"])
    y_min = float(rect["yMin"])
    x_max = x_min + float(rect["width"])
    y_max = y_min + float(rect["height"])
    return x_min, y_min, x_max, y_max

def obj_to_bbox(obj):
    """
    Ưu tiên rectMask nếu có, fallback sang content (polygon).
    Trả về (xmin, ymin, xmax, ymax) hoặc None nếu thiếu.
    """
    if "rectMask" in obj and obj["rectMask"]:
        r = obj["rectMask"]
        if all(k in r for k in ("xMin", "yMin", "width", "height")):
            return rectmask_to_bbox(r)

    if "content" in obj and isinstance(obj["content"], list) and len(obj["content"]) >= 2:
        try:
            return polygon_to_bbox(obj["content"])
        except Exception:
            return None
    return None

def parse_json(json_path):
    """
    JSON có thể là list[object] (đúng như file bạn upload).
    Cũng hỗ trợ dict có key 'objects' / 'annotations'...
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        for k in ("objects", "annotations", "items", "shapes", "content", "labels"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        return [data]  # 1 object đơn
    else:
        return []

def convert_and_copy(image_name):
    name, ext = os.path.splitext(image_name)
    if ext.lower() not in IMG_EXTS:
        return

    img_path = os.path.join(RAW_IMG_DIR, image_name)
    json_path = os.path.join(RAW_JSON_DIR, name + ".json")
    out_img_path = os.path.join(OUT_IMG_DIR, image_name)
    out_label_path = os.path.join(OUT_LABEL_DIR, name + ".txt")

    if not os.path.exists(img_path):
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_name}")
        return
    h, w = img.shape[:2]

    # Nếu không có file JSON thì chỉ copy ảnh (negative)
    if not os.path.exists(json_path):
        shutil.copy2(img_path, out_img_path)
        if os.path.exists(out_label_path):
            os.remove(out_label_path)
        print(f"⚠️ Không có JSON: {image_name} → ảnh negative")
        return

    try:
        objects = parse_json(json_path)
    except Exception as e:
        shutil.copy2(img_path, out_img_path)
        print(f"❌ Lỗi đọc JSON {json_path}: {e}")
        return

    lines = []
    skipped_unknown = 0
    skipped_nobbox = 0

    for obj in objects:
        labels = obj.get("labels", {})
        raw_name = labels.get("labelName", None)
        if not raw_name:
            continue

        norm_name = normalize_label(raw_name)
        if norm_name not in name_to_id:
            skipped_unknown += 1
            continue

        bbox = obj_to_bbox(obj)
        if bbox is None:
            skipped_nobbox += 1
            continue

        xmin, ymin, xmax, ymax = bbox
        # Loại bỏ bbox rỗng / âm
        if xmax <= xmin or ymax <= ymin:
            skipped_nobbox += 1
            continue

        # Chuẩn YOLO (normalize)
        x_center = clamp01(((xmin + xmax) / 2.0) / w)
        y_center = clamp01(((ymin + ymax) / 2.0) / h)
        box_w = clamp01((xmax - xmin) / w)
        box_h = clamp01((ymax - ymin) / h)

        class_id = name_to_id[norm_name]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    # Ghi kết quả
    shutil.copy2(img_path, out_img_path)
    if lines:
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        note = ""
        if skipped_unknown:
            note += f", bỏ {skipped_unknown} nhãn ngoài data.yaml"
        if skipped_nobbox:
            note += f", bỏ {skipped_nobbox} ô không hợp lệ"
        print(f"✅ {image_name}: {len(lines)} box → ghi OK{note}")
    else:
        if os.path.exists(out_label_path):
            os.remove(out_label_path)
        print(f"⚠️ {image_name}: không có box hợp lệ → chỉ copy ảnh")

def process_all():
    files = [f for f in os.listdir(RAW_IMG_DIR) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    files.sort()
    for fname in files:
        convert_and_copy(fname)

if __name__ == "__main__":
    # Cảnh báo mismatch nếu có
    if not names:
        print("❌ data.yaml không có 'names' — hãy kiểm tra lại.")
    else:
        print(f"Loaded {len(names)} classes từ data.yaml")
        process_all()
