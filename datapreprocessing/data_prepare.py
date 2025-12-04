"""
Tạo K-fold cross-validation splits từ dataset

Script này chia dataset thành K folds để thực hiện cross-validation.
Mỗi fold sẽ có thư mục train và val riêng, cùng với file config YAML tương ứng.

Cấu trúc input:
    dataset/train/
        images/          # Tất cả ảnh training
        labels/          # Tất cả labels tương ứng

Cấu trúc output:
    dataset/
        images/
            train_fold1/ # Ảnh training cho fold 1
            val_fold1/   # Ảnh validation cho fold 1
            train_fold2/
            val_fold2/
            ...
        labels/
            train_fold1/ # Labels training cho fold 1
            val_fold1/   # Labels validation cho fold 1
            train_fold2/
            val_fold2/
            ...
        config_fold1.yaml  # Config YOLO cho fold 1
        config_fold2.yaml  # Config YOLO cho fold 2
        ...
"""

import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import KFold
import yaml
from typing import List, Tuple


# ==================== CẤU HÌNH ====================
# Thư mục chứa ảnh và labels gốc
IMAGE_DIR = "dataset/train/images"
LABEL_DIR = "dataset/train/labels"

# Thư mục output
OUTPUT_DIR = "dataset"

# Số lượng folds cho K-fold cross-validation
NUM_FOLDS = 5

# Random seed để đảm bảo reproducibility
RANDOM_SEED = 42

# Danh sách tên class (theo thứ tự ID)
# Có thể thay đổi theo dataset của bạn
CLASS_NAMES = [
    "Sĩ đen",      # 0 - black-advisor
    "Pháo đen",    # 1 - black-cannon
    "Xe đen",      # 2 - black-chariot
    "Tượng đen",   # 3 - black-elephant
    "Tướng đen",   # 4 - black-general
    "Mã đen",      # 5 - black-horse
    "Tốt đen",     # 6 - black-soldier
    "Giao điểm",   # 7 - intersection (nếu có)
    "Sĩ đỏ",       # 8 - red-advisor
    "Pháo đỏ",     # 9 - red-cannon
    "Xe đỏ",       # 10 - red-chariot
    "Tượng đỏ",    # 11 - red-elephant
    "Tướng đỏ",    # 12 - red-general
    "Mã đỏ",       # 13 - red-horse
    "Tốt đỏ"       # 14 - red-soldier
]


# ==================== HÀM XỬ LÝ ====================

def get_image_files(image_dir: str) -> List[str]:
    """
    Lấy danh sách tất cả file ảnh trong thư mục
    
    Args:
        image_dir: Đường dẫn thư mục chứa ảnh
        
    Returns:
        Danh sách tên file ảnh đã được sắp xếp và shuffle
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {image_dir}")
    
    # Lấy danh sách file ảnh
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]
    
    # Sắp xếp và shuffle với seed cố định
    image_files.sort()
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    
    return image_files


def copy_files(files: List[str], split: str, fold: int, 
               image_dir: str, label_dir: str, output_dir: str) -> None:
    """
    Copy ảnh và labels từ thư mục gốc sang thư mục fold
    
    Args:
        files: Danh sách tên file ảnh
        split: 'train' hoặc 'val'
        fold: Số thứ tự fold (1, 2, ...)
        image_dir: Thư mục ảnh gốc
        label_dir: Thư mục labels gốc
        output_dir: Thư mục output
    """
    img_dst_dir = os.path.join(output_dir, f"images/{split}_fold{fold}")
    lbl_dst_dir = os.path.join(output_dir, f"labels/{split}_fold{fold}")
    
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)
    
    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        
        # Đường dẫn nguồn
        img_src = os.path.join(image_dir, img_file)
        label_src = os.path.join(label_dir, label_file)
        
        # Đường dẫn đích
        img_dst = os.path.join(img_dst_dir, img_file)
        label_dst = os.path.join(lbl_dst_dir, label_file)
        
        # Copy ảnh
        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
        
        # Copy label (nếu có)
        if os.path.exists(label_src):
            shutil.copy2(label_src, label_dst)


def create_yaml_config(output_dir: str, fold: int, class_names: List[str]) -> None:
    """
    Tạo file config YAML cho YOLOv8
    
    Args:
        output_dir: Thư mục output
        fold: Số thứ tự fold
        class_names: Danh sách tên class
    """
    config = {
        "path": os.path.abspath(output_dir),
        "train": f"images/train_fold{fold}",
        "val": f"images/val_fold{fold}",
        "nc": len(class_names),  # Số lượng class
        "names": class_names      # Danh sách tên class
    }
    
    yaml_path = os.path.join(output_dir, f"config_fold{fold}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"   📝 Đã tạo: {yaml_path}")


def create_kfold_splits() -> None:
    """
    Tạo K-fold cross-validation splits
    """
    print("🔄 Bắt đầu tạo K-fold splits...\n")
    
    # Lấy danh sách ảnh
    print(f"📂 Đang đọc danh sách ảnh từ: {IMAGE_DIR}")
    image_files = get_image_files(IMAGE_DIR)
    print(f"✅ Tìm thấy {len(image_files)} ảnh\n")
    
    # Kiểm tra thư mục labels
    if not os.path.exists(LABEL_DIR):
        print(f"⚠️  Cảnh báo: Không tìm thấy thư mục labels: {LABEL_DIR}")
    
    # Tạo KFold splitter
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Xử lý từng fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files), 1):
        print(f"📦 Fold {fold}/{NUM_FOLDS}:")
        
        # Chia file thành train và val
        train_files = [image_files[i] for i in train_idx]
        val_files = [image_files[i] for i in val_idx]
        
        print(f"   Train: {len(train_files)} ảnh")
        print(f"   Val: {len(val_files)} ảnh")
        
        # Copy files cho train
        copy_files(train_files, "train", fold, IMAGE_DIR, LABEL_DIR, OUTPUT_DIR)
        
        # Copy files cho val
        copy_files(val_files, "val", fold, IMAGE_DIR, LABEL_DIR, OUTPUT_DIR)
        
        # Tạo file YAML config
        create_yaml_config(OUTPUT_DIR, fold, CLASS_NAMES)
        
        print(f"   ✅ Hoàn thành fold {fold}\n")
    
    print(f"✅ Đã tạo {NUM_FOLDS} folds tại: {OUTPUT_DIR}")
    print(f"\n📊 Cấu trúc thư mục:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── images/")
    print(f"   │   ├── train_fold1/")
    print(f"   │   ├── val_fold1/")
    print(f"   │   ├── train_fold2/")
    print(f"   │   └── ...")
    print(f"   ├── labels/")
    print(f"   │   ├── train_fold1/")
    print(f"   │   ├── val_fold1/")
    print(f"   │   └── ...")
    print(f"   └── config_fold1.yaml, config_fold2.yaml, ...")


# ==================== MAIN ====================

if __name__ == "__main__":
    create_kfold_splits()
