"""
Chuyển đổi annotations từ JSON (Roboflow format) sang YOLO detection format (.txt)

Dataset hỗ trợ:
- https://universe.roboflow.com/chinese-chess/chinese-zyx60
- https://universe.roboflow.com/viktor-ng/chinese-chess-rtpmq

Cấu trúc thư mục mong đợi:
    rawdata/
        images/          # Ảnh gốc từ Roboflow
        labels/          # File JSON annotations từ Roboflow
    Dataset_Detection/piece/Dataset_combined/
        data.yaml        # File cấu hình YOLO với danh sách class names
    processed_data_detection/
        images/          # Ảnh đã xử lý (output)
        labels/          # File .txt YOLO format (output)
"""

import os
import shutil
import json
import cv2
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


# ==================== CẤU HÌNH ====================
# Đường dẫn gốc của project (thay đổi theo môi trường của bạn)
ROOT = r""

# Đường dẫn đến file data.yaml chứa định nghĩa các class
DATA_YAML = os.path.join(ROOT, "Dataset_Detection/piece/Dataset_combined/data.yaml")

# Thư mục chứa dữ liệu thô từ Roboflow
RAW_IMG_DIR = os.path.join(ROOT, "rawdata", "images")
RAW_JSON_DIR = os.path.join(ROOT, "rawdata", "labels")

# Thư mục output cho dữ liệu đã xử lý
OUT_IMG_DIR = os.path.join(ROOT, "processed_data_detection", "images")
OUT_LABEL_DIR = os.path.join(ROOT, "processed_data_detection", "labels")

# Các định dạng ảnh được hỗ trợ
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bảng chuẩn hóa tên class (ví dụ: "guard" -> "advisor")
NAME_NORMALIZERS = {
    "guard": "advisor",
}


# ==================== HÀM TIỆN ÍCH ====================

def clamp01(x: float) -> float:
    """
    Giới hạn giá trị trong khoảng [0.0, 1.0]
    
    Args:
        x: Giá trị cần giới hạn
        
    Returns:
        Giá trị đã được clamp về [0.0, 1.0]
    """
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def normalize_label(name: str) -> str:
    """
    Chuẩn hóa tên class label (ví dụ: "guard" -> "advisor")
    
    Args:
        name: Tên class gốc
        
    Returns:
        Tên class đã được chuẩn hóa
    """
    n = name.strip()
    for src, dst in NAME_NORMALIZERS.items():
        n = n.replace(src, dst)
    return n


def polygon_to_bbox(polygon: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
    """
    Chuyển đổi polygon (danh sách điểm) sang bounding box
    
    Args:
        polygon: Danh sách các điểm có dạng [{"x": ..., "y": ...}, ...]
        
    Returns:
        Tuple (x_min, y_min, x_max, y_max) trong tọa độ pixel
    """
    xs = [pt["x"] for pt in polygon]
    ys = [pt["y"] for pt in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def rectmask_to_bbox(rect: Dict[str, float]) -> Tuple[float, float, float, float]:
    """
    Chuyển đổi rectMask (hình chữ nhật) sang bounding box
    
    Args:
        rect: Dictionary có keys: xMin, yMin, width, height
        
    Returns:
        Tuple (x_min, y_min, x_max, y_max) trong tọa độ pixel
    """
    x_min = float(rect["xMin"])
    y_min = float(rect["yMin"])
    x_max = x_min + float(rect["width"])
    y_max = y_min + float(rect["height"])
    return x_min, y_min, x_max, y_max


def obj_to_bbox(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Trích xuất bounding box từ object annotation
    
    Ưu tiên rectMask nếu có, nếu không thì dùng content (polygon).
    
    Args:
        obj: Object annotation từ JSON
        
    Returns:
        Tuple (x_min, y_min, x_max, y_max) hoặc None nếu không hợp lệ
    """
    # Thử dùng rectMask trước (hình chữ nhật)
    if "rectMask" in obj and obj["rectMask"]:
        r = obj["rectMask"]
        if all(k in r for k in ("xMin", "yMin", "width", "height")):
            return rectmask_to_bbox(r)
    
    # Fallback sang content (polygon)
    if "content" in obj and isinstance(obj["content"], list) and len(obj["content"]) >= 2:
        try:
            return polygon_to_bbox(obj["content"])
        except Exception:
            return None
    
    return None


def parse_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Đọc và parse file JSON annotation từ Roboflow
    
    Hỗ trợ nhiều format JSON khác nhau:
    - List trực tiếp: [obj1, obj2, ...]
    - Dict với key: objects, annotations, items, shapes, content, labels
    
    Args:
        json_path: Đường dẫn đến file JSON
        
    Returns:
        Danh sách các object annotations
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Nếu là list trực tiếp
    if isinstance(data, list):
        return data
    
    # Nếu là dict, tìm key chứa list
    elif isinstance(data, dict):
        for k in ("objects", "annotations", "items", "shapes", "content", "labels"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        # Nếu không tìm thấy, coi như 1 object đơn
        return [data]
    
    else:
        return []


# ==================== HÀM XỬ LÝ CHÍNH ====================

def convert_and_copy(image_name: str, name_to_id: Dict[str, int], img_w: int, img_h: int) -> None:
    """
    Chuyển đổi 1 ảnh và annotations từ JSON sang YOLO format
    
    Args:
        image_name: Tên file ảnh
        name_to_id: Dictionary mapping từ tên class sang ID
        img_w: Chiều rộng ảnh
        img_h: Chiều cao ảnh
    """
    name, ext = os.path.splitext(image_name)
    if ext.lower() not in IMG_EXTS:
        return
    
    # Đường dẫn các file
    img_path = os.path.join(RAW_IMG_DIR, image_name)
    json_path = os.path.join(RAW_JSON_DIR, name + ".json")
    out_img_path = os.path.join(OUT_IMG_DIR, image_name)
    out_label_path = os.path.join(OUT_LABEL_DIR, name + ".txt")
    
    # Kiểm tra ảnh tồn tại
    if not os.path.exists(img_path):
        print(f"⚠️ Không tìm thấy ảnh: {image_name}")
        return
    
    # Nếu không có JSON, chỉ copy ảnh (negative sample)
    if not os.path.exists(json_path):
        shutil.copy2(img_path, out_img_path)
        if os.path.exists(out_label_path):
            os.remove(out_label_path)
        print(f"⚠️ Không có JSON: {image_name} → ảnh negative")
        return
    
    # Đọc và parse JSON
    try:
        objects = parse_json(json_path)
    except Exception as e:
        shutil.copy2(img_path, out_img_path)
        print(f"❌ Lỗi đọc JSON {json_path}: {e}")
        return
    
    # Chuyển đổi từng object sang YOLO format
    lines = []
    skipped_unknown = 0
    skipped_nobbox = 0
    
    for obj in objects:
        # Lấy tên class từ labels
        labels = obj.get("labels", {})
        raw_name = labels.get("labelName", None)
        if not raw_name:
            continue
        
        # Chuẩn hóa tên class
        norm_name = normalize_label(raw_name)
        if norm_name not in name_to_id:
            skipped_unknown += 1
            continue
        
        # Lấy bounding box
        bbox = obj_to_bbox(obj)
        if bbox is None:
            skipped_nobbox += 1
            continue
        
        xmin, ymin, xmax, ymax = bbox
        
        # Kiểm tra bbox hợp lệ
        if xmax <= xmin or ymax <= ymin:
            skipped_nobbox += 1
            continue
        
        # Chuyển đổi sang YOLO format (normalized center x, center y, width, height)
        x_center = clamp01(((xmin + xmax) / 2.0) / img_w)
        y_center = clamp01(((ymin + ymax) / 2.0) / img_h)
        box_w = clamp01((xmax - xmin) / img_w)
        box_h = clamp01((ymax - ymin) / img_h)
        
        # Lấy class ID
        class_id = name_to_id[norm_name]
        
        # Format: class_id x_center y_center width height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    
    # Ghi kết quả
    shutil.copy2(img_path, out_img_path)
    
    if lines:
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        
        note = ""
        if skipped_unknown:
            note += f", bỏ {skipped_unknown} nhãn không có trong data.yaml"
        if skipped_nobbox:
            note += f", bỏ {skipped_nobbox} bbox không hợp lệ"
        print(f"✅ {image_name}: {len(lines)} box → ghi OK{note}")
    else:
        # Nếu không có box hợp lệ, xóa file label
        if os.path.exists(out_label_path):
            os.remove(out_label_path)
        print(f"⚠️ {image_name}: không có box hợp lệ → chỉ copy ảnh")


def process_all() -> None:
    """
    Xử lý tất cả ảnh trong thư mục rawdata/images
    """
    # Đọc data.yaml để lấy danh sách class
    if not os.path.exists(DATA_YAML):
        print(f"❌ Không tìm thấy file data.yaml tại: {DATA_YAML}")
        print("   Hãy tạo file data.yaml với cấu trúc:")
        print("   names: ['black-advisor', 'black-cannon', ...]")
        return
    
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    
    names = yaml_data.get("names", [])
    if not names:
        print("❌ data.yaml không có 'names' — hãy kiểm tra lại.")
        return
    
    # Tạo mapping từ tên class sang ID
    name_to_id = {n: i for i, n in enumerate(names)}
    print(f"📋 Đã load {len(names)} classes từ data.yaml:")
    for name, id_val in name_to_id.items():
        print(f"   {id_val}: {name}")
    
    # Tạo thư mục output
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)
    
    # Lấy danh sách tất cả ảnh
    if not os.path.exists(RAW_IMG_DIR):
        print(f"❌ Không tìm thấy thư mục: {RAW_IMG_DIR}")
        return
    
    image_files = [
        f for f in os.listdir(RAW_IMG_DIR)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]
    image_files.sort()
    
    print(f"\n🔄 Bắt đầu xử lý {len(image_files)} ảnh...\n")
    
    # Xử lý từng ảnh
    for image_name in image_files:
        img_path = os.path.join(RAW_IMG_DIR, image_name)
        
        # Đọc kích thước ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Không đọc được ảnh: {image_name}")
            continue
        
        h, w = img.shape[:2]
        convert_and_copy(image_name, name_to_id, w, h)
    
    print(f"\n✅ Hoàn thành! Dữ liệu đã được lưu tại:")
    print(f"   Images: {OUT_IMG_DIR}")
    print(f"   Labels: {OUT_LABEL_DIR}")


# ==================== MAIN ====================

if __name__ == "__main__":
    process_all()
