import os
import shutil
import yaml
from pathlib import Path

# === Nhập: đường dẫn tới file YAML gốc
original_yaml_path = r"Dataset\piece\Dataset4_270image\data.yaml"

# === Bảng chuẩn: tên nhãn → ID mới
label_map = {
    'black-advisor': 100, 'red-advisor': 101,
    'black-elephant': 102, 'red-elephant': 103,
    'black-cannon': 104, 'red-cannon': 105,
    'black-chariot': 106, 'red-chariot': 107,
    'black-general': 108, 'red-general': 109,
    'black-horse': 110, 'red-horse': 111,
    'black-soldier': 112, 'red-soldier': 113
}

# === Đọc YAML
with open(original_yaml_path, 'r') as f:
    yaml_data = yaml.safe_load(f)

# === Chuẩn hóa tên nhãn từ YAML
original_names = [name.replace("guard", "advisor") for name in yaml_data['names']]
oldid_to_name = {i: original_names[i] for i in range(len(original_names))}
oldid_to_newid = {
    i: label_map[name]
    for i, name in oldid_to_name.items()
    if name in label_map
}

# === Đường dẫn gốc
dataset_dir = Path(original_yaml_path).parent.resolve()

# === Tạo thư mục mới
output_dir = dataset_dir / "remapped_dts"
if output_dir.exists():
    shutil.rmtree(output_dir)
shutil.copytree(dataset_dir, output_dir)

# === Danh sách các thư mục cần xử lý
split_folders = ['train', 'val', 'test']
for split in split_folders:
    label_path = output_dir / split / "labels"
    if not label_path.exists():
        continue

    for filename in os.listdir(label_path):
        if not filename.endswith(".txt"):
            continue

        file_path = label_path / filename
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            if old_id in oldid_to_newid:
                parts[0] = str(oldid_to_newid[old_id])
                new_lines.append(' '.join(parts))

        with open(file_path, 'w') as f:
            if new_lines:
                f.write('\n'.join(new_lines) + '\n')
            else:
                f.write('')  # ghi rỗng nếu không hợp lệ

# === Ghi lại YAML mới (nếu cần)
remapped_yaml = {
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images' if 'test' in yaml_data else '',
    'nc': len(label_map),
    'names': list(label_map.keys())
}

with open(output_dir / "remapped.yaml", 'w') as f:
    yaml.dump(remapped_yaml, f)

print(f"✅ Đã tạo bản sao dataset với ID mới tại: {output_dir}")
