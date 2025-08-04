import os
import random
import shutil
from sklearn.model_selection import KFold
import yaml

# === CẤU HÌNH ===
image_dir = "dataset/train/images"
label_dir = "dataset/train/labels"
output_dir = "dataset"
num_folds = 5  # bạn có thể đổi K fold ở đây
random_seed = 42

# === DANH SÁCH LỚP TIẾNG VIỆT ===
class_names = [
    "Sĩ đen", "Pháo đen", "Xe đen", "Tượng đen", "Tướng đen", "Mã đen", "Tốt đen",
    "Giao điểm",
    "Sĩ đỏ", "Pháo đỏ", "Xe đỏ", "Tượng đỏ", "Tướng đỏ", "Mã đỏ", "Tốt đỏ"
]

# === LẤY DANH SÁCH ẢNH ===
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
image_files.sort()
random.seed(random_seed)
random.shuffle(image_files)

# === TẠO K-FOLD ===
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_files), 1):
    print(f"\n📂 Fold {fold}:")
    for split in ['train', 'val']:
        os.makedirs(f"{output_dir}/images/{split}_fold{fold}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}_fold{fold}", exist_ok=True)

    train_files = [image_files[i] for i in train_idx]
    val_files = [image_files[i] for i in val_idx]

    def copy_files(files, split):
        for img_file in files:
            base = os.path.splitext(img_file)[0]
            label_file = base + ".txt"

            img_src = os.path.join(image_dir, img_file)
            label_src = os.path.join(label_dir, label_file)

            img_dst = os.path.join(output_dir, f"images/{split}_fold{fold}", img_file)
            label_dst = os.path.join(output_dir, f"labels/{split}_fold{fold}", label_file)

            if os.path.exists(label_src):
                shutil.copyfile(img_src, img_dst)
                shutil.copyfile(label_src, label_dst)

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print(f"✅ Train: {len(train_files)} ảnh, Val: {len(val_files)} ảnh")

    # === TẠO FILE config.yaml CHO YOLOv8 ===
    config = {
        "path": "./dataset",
        "train": f"images/train_fold{fold}",
        "val": f"images/val_fold{fold}",
        "nc": len(class_names),
        "names": class_names
    }

    yaml_path = os.path.join(output_dir, f"config_fold{fold}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    print(f"📝 Đã tạo: {yaml_path}")
