import os
import shutil
from pathlib import Path
import yaml

# === Cấu hình ===
remap_dirs = [
    Path("Dataset/piece/remapped_dts1"),
    Path("Dataset/piece/remapped_dts2")
]
output_dir = Path("Dataset/piece/Dataset_combined")

# === Bảng chuyển từ ID tạm (100–113) → chuẩn 0–13 ===
id_map = {
    100: 0,   104: 1,   106: 2,   102: 3,
    108: 4,   110: 5,   112: 6,   101: 7,
    105: 8,   107: 9,   103: 10,  109: 11,
    111: 12,  113: 13
}

# === Class name theo thứ tự chuẩn ===
standard_names = [
    'black-advisor', 'black-cannon', 'black-chariot', 'black-elephant',
    'black-general', 'black-horse', 'black-soldier',
    'red-advisor', 'red-cannon', 'red-chariot', 'red-elephant',
    'red-general', 'red-horse', 'red-soldier'
]

# === Các thư mục cần xử lý
splits = ['train', 'val', 'test']

# === Bắt đầu gộp
if output_dir.exists():
    shutil.rmtree(output_dir)
for split in splits:
    (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

# === Duyệt từng dataset và copy từng ảnh/label
for src_dir in remap_dirs:
    for split in splits:
        src_img_dir = src_dir / split / 'images'
        src_lbl_dir = src_dir / split / 'labels'

        if not src_img_dir.exists() or not src_lbl_dir.exists():
            continue

        for file in os.listdir(src_img_dir):
            src_img = src_img_dir / file
            dst_img = output_dir / split / 'images' / file
            shutil.copy2(src_img, dst_img)

        for file in os.listdir(src_lbl_dir):
            if not file.endswith(".txt"):
                continue
            src_lbl = src_lbl_dir / file
            dst_lbl = output_dir / split / 'labels' / file

            with open(src_lbl, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_id = int(parts[0])
                if old_id in id_map:
                    parts[0] = str(id_map[old_id])
                    new_lines.append(' '.join(parts))

            with open(dst_lbl, 'w') as f:
                if new_lines:
                    f.write('\n'.join(new_lines) + '\n')
                else:
                    f.write('')  # nếu không còn nhãn hợp lệ

# === Tạo YAML mới
yaml_path = output_dir / "data.yaml"
yaml_data = {
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': len(standard_names),
    'names': standard_names
}

with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f)

print(f"✅ Dataset đã gộp và chuẩn hóa tại: {output_dir}")
