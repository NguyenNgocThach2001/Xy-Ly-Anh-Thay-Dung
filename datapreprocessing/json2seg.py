"""
Chuyển đổi annotations từ JSON (Roboflow format) sang YOLO segmentation format (.txt)

Script này dùng để tạo dataset cho board detection (phát hiện bàn cờ).
Mỗi ảnh sẽ có 1 polygon đại diện cho 4 góc của bàn cờ.

Dataset hỗ trợ:
- https://universe.roboflow.com/chinese-chess/chinese-zyx60
- https://universe.roboflow.com/viktor-ng/chinese-chess-rtpmq

Cấu trúc thư mục mong đợi:
    rawdata/
        images/          # Ảnh gốc từ Roboflow
        labels/          # File JSON annotations từ Roboflow
    seg_data/
        images/          # Ảnh đã xử lý (output)
        labels/          # File .txt YOLO segmentation format (output)
"""

import os
import shutil
import json
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional


# ==================== CẤU HÌNH ====================
# Đường dẫn gốc của project (thay đổi theo môi trường của bạn)
ROOT = r""

# Thư mục chứa dữ liệu thô từ Roboflow
RAW_IMG_DIR = os.path.join(ROOT, "rawdata", "images")
RAW_JSON_DIR = os.path.join(ROOT, "rawdata", "labels")

# Thư mục output cho dữ liệu segmentation
OUT_IMG_DIR = os.path.join(ROOT, "seg_data", "images")
OUT_LABEL_DIR = os.path.join(ROOT, "seg_data", "labels")

# Các định dạng ảnh được hỗ trợ
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class ID cho board (thường là 0 vì chỉ có 1 class)
CLASS_ID = 0


# ==================== HÀM TIỆN ÍCH ====================

def parse_json(json_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Đọc và parse file JSON annotation từ Roboflow
    
    Args:
        json_path: Đường dẫn đến file JSON
        
    Returns:
        Danh sách các object annotations hoặc None nếu lỗi
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
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
        
        return None
    except Exception as e:
        print(f"❌ Lỗi đọc JSON {json_path}: {e}")
        return None


def extract_board_polygon(obj: Dict[str, Any]) -> Optional[List[Dict[str, float]]]:
    """
    Trích xuất polygon của bàn cờ từ object annotation
    
    Args:
        obj: Object annotation từ JSON
        
    Returns:
        Danh sách các điểm polygon hoặc None nếu không hợp lệ
    """
    # Tìm polygon trong content
    if "content" in obj and isinstance(obj["content"], list):
        content = obj["content"]
        # Kiểm tra có đủ ít nhất 4 điểm (tứ giác)
        if len(content) >= 4:
            # Kiểm tra format của các điểm
            if all(isinstance(pt, dict) and "x" in pt and "y" in pt for pt in content):
                return content
    
    return None


# ==================== HÀM XỬ LÝ CHÍNH ====================

def convert_and_copy(image_name: str) -> None:
    """
    Chuyển đổi 1 ảnh và annotations từ JSON sang YOLO segmentation format
    
    Format YOLO segmentation:
        class_id x1 y1 x2 y2 x3 y3 x4 y4 ...
    (tọa độ được normalize về [0, 1])
    
    Args:
        image_name: Tên file ảnh
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
    
    # Đọc ảnh để lấy kích thước
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_name}")
        return
    
    h, w = img.shape[:2]
    
    # Kiểm tra JSON tồn tại
    if not os.path.exists(json_path):
        print(f"⚠️ Không tìm thấy JSON cho {image_name}")
        return
    
    # Đọc và parse JSON
    objects = parse_json(json_path)
    if not objects:
        print(f"⚠️ JSON không hợp lệ hoặc rỗng: {json_path}")
        return
    
    # Tìm polygon của bàn cờ (thường là object đầu tiên)
    board_polygon = None
    for obj in objects:
        polygon = extract_board_polygon(obj)
        if polygon:
            board_polygon = polygon
            break
    
    if not board_polygon:
        print(f"⚠️ Không tìm thấy polygon hợp lệ trong JSON: {json_path}")
        return
    
    # Chuyển đổi tọa độ sang normalized [0, 1]
    coords = []
    for pt in board_polygon:
        x = pt["x"] / w  # Normalize theo chiều rộng
        y = pt["y"] / h  # Normalize theo chiều cao
        coords.extend([x, y])
    
    # Ghi file label .txt theo format YOLO segmentation
    with open(out_label_path, "w", encoding="utf-8") as out:
        # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 ...
        coord_str = " ".join(f"{c:.6f}" for c in coords)
        out.write(f"{CLASS_ID} {coord_str}\n")
    
    # Copy ảnh
    shutil.copy2(img_path, out_img_path)
    print(f"✅ {image_name}: {len(board_polygon)} điểm → đã xử lý và lưu")


def process_all() -> None:
    """
    Xử lý tất cả ảnh trong thư mục rawdata/images
    """
    # Tạo thư mục output
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)
    
    # Kiểm tra thư mục input
    if not os.path.exists(RAW_IMG_DIR):
        print(f"❌ Không tìm thấy thư mục: {RAW_IMG_DIR}")
        return
    
    # Lấy danh sách tất cả ảnh
    image_files = [
        f for f in os.listdir(RAW_IMG_DIR)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]
    image_files.sort()
    
    print(f"🔄 Bắt đầu xử lý {len(image_files)} ảnh cho segmentation...\n")
    
    # Xử lý từng ảnh
    for image_name in image_files:
        convert_and_copy(image_name)
    
    print(f"\n✅ Hoàn thành! Dữ liệu segmentation đã được lưu tại:")
    print(f"   Images: {OUT_IMG_DIR}")
    print(f"   Labels: {OUT_LABEL_DIR}")


# ==================== MAIN ====================

if __name__ == "__main__":
    process_all()
