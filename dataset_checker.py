import os
import cv2
import yaml
import numpy as np
from math import ceil

# ========== CẤU HÌNH ==========
images_dir = r"Dataset\piece\Dataset5_64image_nghieng\train\images"
labels_dir = r"Dataset\piece\Dataset5_64image_nghieng\train\labels"
data_yaml  = r"Dataset\piece\Dataset5_64image_nghieng\data.yaml"

BATCH_SIZE = 6           # số ảnh mỗi batch
GRID_COLS  = 3           # số cột trong lưới
MAX_CELL_W = 640         # giới hạn chiều rộng mỗi ảnh
MAX_CELL_H = 640         # giới hạn chiều cao mỗi ảnh

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 5.0         # tăng từ 0.6 lên 1.0
THICKNESS   = 2           # độ dày chữ
BOX_COLOR   = (36, 255, 12)
TEXT_COLOR  = (0, 0, 255)
FNAME_COLOR = (255, 255, 255)

# ========== Đọc names từ data.yaml ==========
with open(data_yaml, "r", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)
NAMES = data_cfg.get("names", [])
def get_label_name(cid: int) -> str:
    return NAMES[cid] if 0 <= cid < len(NAMES) else str(cid)

# ========== Utils ==========
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def draw_one_image_with_labels(img_path: str, label_path: str):
    img = cv2.imread(img_path)
    if img is None:
        err = np.full((320, 480, 3), 32, np.uint8)
        cv2.putText(err, "Cannot read image", (20, 160), FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return err

    h, w = img.shape[:2]
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, box_w, box_h = map(float, parts)
                cx, cy = x_center * w, y_center * h
                bw, bh = box_w * w, box_h * h
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

                # Vẽ box
                cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

                # Nhãn (to hơn)
                label_text = get_label_name(int(class_id))
                (tw, th), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, THICKNESS)
                # nền chữ để dễ đọc
                cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), BOX_COLOR, -1)
                cv2.putText(img, label_text, (x1 + 2, y1 - baseline - 2),
                            FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)

    # Tên file ở góc trên
    fname = os.path.basename(img_path)
    cv2.putText(img, fname, (10, 30), FONT, FONT_SCALE, FNAME_COLOR, THICKNESS, cv2.LINE_AA)

    # Resize về cell
    ih, iw = img.shape[:2]
    scale = min(MAX_CELL_W / iw, MAX_CELL_H / ih, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)
    return img

def make_grid(images):
    cols = GRID_COLS
    rows = ceil(len(images) / cols)
    cell_h, cell_w = MAX_CELL_H, MAX_CELL_W
    pad_color = (30, 30, 30)

    canvas_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(images):
                im = images[idx]
                h, w = im.shape[:2]
                cell = np.full((cell_h, cell_w, 3), pad_color, np.uint8)
                cell[0:h, 0:w] = im
            else:
                cell = np.full((cell_h, cell_w, 3), pad_color, np.uint8)
            row_imgs.append(cell)
        canvas_rows.append(np.hstack(row_imgs))

    grid = np.vstack(canvas_rows)
    cv2.rectangle(grid, (0, 0), (grid.shape[1]-1, 40), (0, 0, 0), -1)
    cv2.putText(grid, "Press 1: prev | 2: next | q/ESC: quit",
                (10, 28), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)
    return grid

# ========== Chuẩn bị ảnh ==========
image_files = sorted([f for f in os.listdir(images_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
if not image_files:
    raise FileNotFoundError("Không tìm thấy ảnh trong images_dir")

batch_idx = 0
num_batches = ceil(len(image_files) / BATCH_SIZE)

while True:
    start = batch_idx * BATCH_SIZE
    end = min(len(image_files), start + BATCH_SIZE)
    batch_files = image_files[start:end]

    tiles = [draw_one_image_with_labels(
        os.path.join(images_dir, f),
        os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")
    ) for f in batch_files]

    grid = make_grid(tiles)
    win_name = f"Batch {batch_idx+1}/{num_batches}  ({start+1}-{end}/{len(image_files)})"
    cv2.imshow(win_name, grid)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win_name)

    if key in (ord('q'), 27):
        break
    elif key == ord('1'):
        batch_idx = (batch_idx - 1) % num_batches
    elif key == ord('2'):
        batch_idx = (batch_idx + 1) % num_batches
