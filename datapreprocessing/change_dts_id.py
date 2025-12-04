"""
Chuẩn hóa ID của các class trong dataset

Script này đọc dataset với các class ID gốc và chuyển đổi sang ID chuẩn hóa (100-113).
Điều này giúp chuẩn bị dataset trước khi merge với dataset khác.

Quy trình:
1. Đọc file data.yaml gốc để lấy mapping tên class -> ID cũ
2. Áp dụng bảng mapping ID cũ -> ID mới (100-113)
3. Cập nhật tất cả file label .txt trong train/val/test
4. Tạo file data.yaml mới với ID đã chuẩn hóa

Cấu trúc dataset:
    Dataset_Detection/piece/Dataset4_270image/
        data.yaml              # File cấu hình gốc
        train/
            images/
            labels/             # File .txt với ID cũ
        val/
            images/
            labels/
        test/
            images/
            labels/
    Dataset_Detection/piece/Dataset4_270image/remapped_dts/
        remapped.yaml          # File cấu hình mới (output)
        train/
            images/
            labels/             # File .txt với ID mới (output)
        val/
            images/
            labels/
        test/
            images/
            labels/
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List


# ==================== CẤU HÌNH ====================
# Đường dẫn đến file YAML gốc của dataset
ORIGINAL_YAML_PATH = r"Dataset_Detection\piece\Dataset4_270image\data.yaml"

# Bảng mapping: tên class -> ID mới (chuẩn hóa)
# ID từ 100-113 để tránh conflict với ID gốc
LABEL_MAP = {
    'black-advisor': 100,
    'red-advisor': 101,
    'black-elephant': 102,
    'red-elephant': 103,
    'black-cannon': 104,
    'red-cannon': 105,
    'black-chariot': 106,
    'red-chariot': 107,
    'black-general': 108,
    'red-general': 109,
    'black-horse': 110,
    'red-horse': 111,
    'black-soldier': 112,
    'red-soldier': 113
}

# Các thư mục split cần xử lý
SPLIT_FOLDERS = ['train', 'val', 'test']


# ==================== HÀM XỬ LÝ ====================

def normalize_class_name(name: str) -> str:
    """
    Chuẩn hóa tên class (ví dụ: "guard" -> "advisor")
    
    Args:
        name: Tên class gốc
        
    Returns:
        Tên class đã chuẩn hóa
    """
    return name.replace("guard", "advisor")


def create_id_mapping(yaml_data: Dict) -> Dict[int, int]:
    """
    Tạo mapping từ ID cũ sang ID mới dựa trên tên class
    
    Args:
        yaml_data: Dữ liệu từ file YAML gốc
        
    Returns:
        Dictionary mapping {old_id: new_id}
    """
    # Lấy danh sách tên class từ YAML và chuẩn hóa
    original_names = yaml_data.get('names', [])
    normalized_names = [normalize_class_name(name) for name in original_names]
    
    # Tạo mapping ID cũ -> tên class
    oldid_to_name = {i: normalized_names[i] for i in range(len(normalized_names))}
    
    # Tạo mapping ID cũ -> ID mới
    oldid_to_newid = {}
    for old_id, name in oldid_to_name.items():
        if name in LABEL_MAP:
            oldid_to_newid[old_id] = LABEL_MAP[name]
    
    return oldid_to_newid


def remap_label_file(label_path: Path, oldid_to_newid: Dict[int, int]) -> None:
    """
    Cập nhật ID trong file label .txt
    
    Format YOLO: class_id x_center y_center width height
    
    Args:
        label_path: Đường dẫn đến file label .txt
        oldid_to_newid: Dictionary mapping ID cũ -> ID mới
    """
    if not label_path.exists():
        return
    
    # Đọc file label
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        
        # Kiểm tra format hợp lệ (phải có 5 giá trị)
        if len(parts) != 5:
            continue
        
        try:
            old_id = int(parts[0])
            
            # Nếu có mapping, thay thế ID
            if old_id in oldid_to_newid:
                parts[0] = str(oldid_to_newid[old_id])
                new_lines.append(' '.join(parts))
        except ValueError:
            # Bỏ qua dòng không hợp lệ
            continue
    
    # Ghi lại file
    with open(label_path, 'w', encoding='utf-8') as f:
        if new_lines:
            f.write('\n'.join(new_lines) + '\n')
        else:
            f.write('')  # Ghi rỗng nếu không còn nhãn hợp lệ


def process_dataset() -> None:
    """
    Xử lý toàn bộ dataset: chuẩn hóa ID và tạo dataset mới
    """
    # Kiểm tra file YAML tồn tại
    original_yaml_path = Path(ORIGINAL_YAML_PATH)
    if not original_yaml_path.exists():
        print(f"❌ Không tìm thấy file YAML: {ORIGINAL_YAML_PATH}")
        return
    
    # Đọc YAML gốc
    print(f"📖 Đang đọc file YAML: {ORIGINAL_YAML_PATH}")
    with open(original_yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    # Tạo mapping ID cũ -> ID mới
    oldid_to_newid = create_id_mapping(yaml_data)
    print(f"📋 Đã tạo mapping cho {len(oldid_to_newid)} class:")
    for old_id, new_id in sorted(oldid_to_newid.items()):
        old_name = yaml_data.get('names', [])[old_id] if old_id < len(yaml_data.get('names', [])) else f"ID_{old_id}"
        print(f"   {old_id} ({old_name}) -> {new_id}")
    
    # Đường dẫn dataset gốc
    dataset_dir = original_yaml_path.parent.resolve()
    print(f"\n📂 Dataset gốc: {dataset_dir}")
    
    # Tạo thư mục output (sao chép toàn bộ dataset)
    output_dir = dataset_dir / "remapped_dts"
    if output_dir.exists():
        print(f"🗑️  Xóa thư mục cũ: {output_dir}")
        shutil.rmtree(output_dir)
    
    print(f"📋 Đang sao chép dataset...")
    shutil.copytree(dataset_dir, output_dir)
    print(f"✅ Đã sao chép dataset sang: {output_dir}")
    
    # Xử lý từng split (train/val/test)
    print(f"\n🔄 Đang cập nhật ID trong các file label...")
    for split in SPLIT_FOLDERS:
        label_path = output_dir / split / "labels"
        
        if not label_path.exists():
            print(f"⚠️  Không tìm thấy thư mục: {label_path}")
            continue
        
        # Đếm số file đã xử lý
        count = 0
        for filename in os.listdir(label_path):
            if not filename.endswith(".txt"):
                continue
            
            file_path = label_path / filename
            remap_label_file(file_path, oldid_to_newid)
            count += 1
        
        print(f"   ✅ {split}: đã xử lý {count} file label")
    
    # Tạo file YAML mới với ID đã chuẩn hóa
    remapped_yaml = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images' if 'test' in yaml_data.get('test', '') or (output_dir / 'test').exists() else '',
        'nc': len(LABEL_MAP),  # Số lượng class
        'names': list(LABEL_MAP.keys())  # Danh sách tên class theo thứ tự ID
    }
    
    yaml_output_path = output_dir / "remapped.yaml"
    with open(yaml_output_path, 'w', encoding='utf-8') as f:
        yaml.dump(remapped_yaml, f, allow_unicode=True, default_flow_style=False)
    
    print(f"\n✅ Đã tạo file YAML mới: {yaml_output_path}")
    print(f"\n✅ Hoàn thành! Dataset đã được chuẩn hóa tại: {output_dir}")
    print(f"   - Tổng số class: {len(LABEL_MAP)}")
    print(f"   - ID range: {min(LABEL_MAP.values())} - {max(LABEL_MAP.values())}")


# ==================== MAIN ====================

if __name__ == "__main__":
    process_dataset()
