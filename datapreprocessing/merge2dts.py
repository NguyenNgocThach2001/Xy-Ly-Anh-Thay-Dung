"""
Gộp 2 dataset và chuẩn hóa ID về range 0-13

Script này gộp 2 dataset đã được chuẩn hóa ID (100-113) từ change_dts_id.py
và chuyển đổi ID về range chuẩn 0-13 để sử dụng với YOLO.

Quy trình:
1. Đọc 2 dataset đã được remap (có ID 100-113)
2. Gộp tất cả ảnh và label từ 2 dataset
3. Chuyển đổi ID từ 100-113 về 0-13
4. Tạo file data.yaml mới với ID chuẩn

Dataset hỗ trợ:
- https://universe.roboflow.com/chinese-chess/chinese-zyx60
- https://universe.roboflow.com/viktor-ng/chinese-chess-rtpmq

Cấu trúc dataset input:
    Dataset_Detection/piece/remapped_dts1/    # Dataset 1 đã remap (ID 100-113)
        train/images/
        train/labels/
        val/images/
        val/labels/
        test/images/
        test/labels/
    Dataset_Detection/piece/remapped_dts2/    # Dataset 2 đã remap (ID 100-113)
        train/images/
        train/labels/
        val/images/
        val/labels/
        test/images/
        test/labels/

Cấu trúc dataset output:
    Dataset_Detection/piece/Dataset_combined/  # Dataset đã gộp (ID 0-13)
        data.yaml
        train/images/
        train/labels/
        val/images/
        val/labels/
        test/images/
        test/labels/
"""

import os
import shutil
from pathlib import Path
import yaml
from typing import Dict, List, Set


# ==================== CẤU HÌNH ====================
# Danh sách các dataset cần gộp (đã được remap với ID 100-113)
REMAP_DIRS = [
    Path("Dataset_Detection/piece/remapped_dts1"),
    Path("Dataset_Detection/piece/remapped_dts2")
]

# Thư mục output cho dataset đã gộp
OUTPUT_DIR = Path("Dataset_Detection/piece/Dataset_combined")

# Bảng mapping từ ID tạm (100-113) sang ID chuẩn (0-13)
# Thứ tự: black-advisor, black-cannon, black-chariot, black-elephant,
#         black-general, black-horse, black-soldier,
#         red-advisor, red-cannon, red-chariot, red-elephant,
#         red-general, red-horse, red-soldier
ID_MAP = {
    100: 0,   # black-advisor -> 0
    104: 1,   # black-cannon -> 1
    106: 2,   # black-chariot -> 2
    102: 3,   # black-elephant -> 3
    108: 4,   # black-general -> 4
    110: 5,   # black-horse -> 5
    112: 6,   # black-soldier -> 6
    101: 7,   # red-advisor -> 7
    105: 8,   # red-cannon -> 8
    107: 9,   # red-chariot -> 9
    103: 10,  # red-elephant -> 10
    109: 11,  # red-general -> 11
    111: 12,  # red-horse -> 12
    113: 13   # red-soldier -> 13
}

# Tên class theo thứ tự ID chuẩn (0-13)
STANDARD_NAMES = [
    'black-advisor',    # 0
    'black-cannon',     # 1
    'black-chariot',    # 2
    'black-elephant',   # 3
    'black-general',    # 4
    'black-horse',      # 5
    'black-soldier',    # 6
    'red-advisor',      # 7
    'red-cannon',       # 8
    'red-chariot',      # 9
    'red-elephant',     # 10
    'red-general',      # 11
    'red-horse',        # 12
    'red-soldier'       # 13
]

# Các thư mục split cần xử lý
SPLITS = ['train', 'val', 'test']


# ==================== HÀM XỬ LÝ ====================

def remap_label_file_content(content: str, id_map: Dict[int, int]) -> List[str]:
    """
    Chuyển đổi ID trong nội dung file label
    
    Args:
        content: Nội dung file label (chuỗi)
        id_map: Dictionary mapping ID cũ -> ID mới
        
    Returns:
        Danh sách các dòng đã được remap
    """
    lines = content.strip().split('\n')
    new_lines = []
    
    for line in lines:
        parts = line.strip().split()
        
        # Kiểm tra format hợp lệ (phải có 5 giá trị cho detection)
        if len(parts) != 5:
            continue
        
        try:
            old_id = int(parts[0])
            
            # Nếu có mapping, thay thế ID
            if old_id in id_map:
                parts[0] = str(id_map[old_id])
                new_lines.append(' '.join(parts))
        except ValueError:
            # Bỏ qua dòng không hợp lệ
            continue
    
    return new_lines


def copy_and_remap_labels(src_label_dir: Path, dst_label_dir: Path, id_map: Dict[int, int]) -> int:
    """
    Copy và remap tất cả file label từ thư mục nguồn sang thư mục đích
    
    Args:
        src_label_dir: Thư mục label nguồn
        dst_label_dir: Thư mục label đích
        id_map: Dictionary mapping ID cũ -> ID mới
        
    Returns:
        Số file đã xử lý
    """
    if not src_label_dir.exists():
        return 0
    
    count = 0
    for filename in os.listdir(src_label_dir):
        if not filename.endswith(".txt"):
            continue
        
        src_label_path = src_label_dir / filename
        dst_label_path = dst_label_dir / filename
        
        # Đọc file nguồn
        with open(src_label_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remap ID
        new_lines = remap_label_file_content(content, id_map)
        
        # Ghi file đích
        with open(dst_label_path, 'w', encoding='utf-8') as f:
            if new_lines:
                f.write('\n'.join(new_lines) + '\n')
            else:
                f.write('')  # Ghi rỗng nếu không còn nhãn hợp lệ
        
        count += 1
    
    return count


def merge_datasets() -> None:
    """
    Gộp các dataset và chuẩn hóa ID
    """
    print("🔄 Bắt đầu gộp dataset...\n")
    
    # Kiểm tra các dataset nguồn
    for i, src_dir in enumerate(REMAP_DIRS, 1):
        if not src_dir.exists():
            print(f"⚠️  Dataset {i} không tồn tại: {src_dir}")
        else:
            print(f"✅ Dataset {i}: {src_dir}")
    
    # Xóa thư mục output cũ nếu có
    if OUTPUT_DIR.exists():
        print(f"\n🗑️  Xóa thư mục output cũ: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    # Tạo cấu trúc thư mục output
    print(f"\n📂 Tạo cấu trúc thư mục output: {OUTPUT_DIR}")
    for split in SPLITS:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Dictionary để track số file đã copy (tránh trùng tên)
    file_counter = {}  # {split: {filename: count}}
    for split in SPLITS:
        file_counter[split] = {}
    
    # Duyệt từng dataset và gộp
    total_images = {split: 0 for split in SPLITS}
    total_labels = {split: 0 for split in SPLITS}
    
    for dataset_idx, src_dir in enumerate(REMAP_DIRS, 1):
        if not src_dir.exists():
            continue
        
        print(f"\n📦 Xử lý dataset {dataset_idx}: {src_dir.name}")
        
        for split in SPLITS:
            src_img_dir = src_dir / split / 'images'
            src_lbl_dir = src_dir / split / 'labels'
            
            if not src_img_dir.exists() or not src_lbl_dir.exists():
                print(f"   ⚠️  Không tìm thấy split '{split}' trong dataset {dataset_idx}")
                continue
            
            # Copy ảnh (xử lý trùng tên bằng cách đổi tên)
            img_count = 0
            for filename in os.listdir(src_img_dir):
                src_img = src_img_dir / filename
                
                # Nếu file đã tồn tại, đổi tên
                dst_img = OUTPUT_DIR / split / 'images' / filename
                if dst_img.exists():
                    name, ext = os.path.splitext(filename)
                    counter = file_counter[split].get(filename, 0) + 1
                    file_counter[split][filename] = counter
                    new_filename = f"{name}_dts{dataset_idx}_{counter}{ext}"
                    dst_img = OUTPUT_DIR / split / 'images' / new_filename
                
                shutil.copy2(src_img, dst_img)
                img_count += 1
            
            # Copy và remap labels
            lbl_count = copy_and_remap_labels(
                src_lbl_dir,
                OUTPUT_DIR / split / 'labels',
                ID_MAP
            )
            
            total_images[split] += img_count
            total_labels[split] += lbl_count
            print(f"   ✅ {split}: {img_count} ảnh, {lbl_count} labels")
    
    # Tạo file YAML mới
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_data = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(STANDARD_NAMES),  # Số lượng class
        'names': STANDARD_NAMES      # Danh sách tên class
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False)
    
    print(f"\n✅ Đã tạo file YAML: {yaml_path}")
    print(f"\n📊 Tổng kết:")
    for split in SPLITS:
        print(f"   {split}: {total_images[split]} ảnh, {total_labels[split]} labels")
    print(f"\n✅ Dataset đã được gộp và chuẩn hóa tại: {OUTPUT_DIR}")
    print(f"   - Tổng số class: {len(STANDARD_NAMES)}")
    print(f"   - ID range: 0 - {len(STANDARD_NAMES) - 1}")


# ==================== MAIN ====================

if __name__ == "__main__":
    merge_datasets()
