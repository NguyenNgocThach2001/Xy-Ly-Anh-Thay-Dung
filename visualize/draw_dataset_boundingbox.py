import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# === CẤU HÌNH ===
image_path = "Dataset/images/train/35c9bb61-Untitled-181_jpg.rf.692b2642760ac2a724387f90b2aae0e5.jpg"
label_path = "Dataset/labels/train/35c9bb61-Untitled-181_jpg.rf.692b2642760ac2a724387f90b2aae0e5.txt"

class_names = [
    "Xe đỏ", "Mã đỏ", "Tượng đỏ", "Sĩ đỏ", "Tướng đỏ", "Pháo đỏ", "Tốt đỏ",
    "Xe đen", "Mã đen", "Tượng đen", "Sĩ đen", "Tướng đen", "Pháo đen", "Tốt đen", "Khác"
]

font_path = "arial.ttf"  # 🔁 Đảm bảo file font có hỗ trợ Unicode, bạn có thể dùng Roboto, Arial Unicode MS, v.v.
font_size = 20

# === ĐỌC ẢNH & CHUYỂN SANG PIL ===
image = cv2.imread(image_path)
h, w = image.shape[:2]
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)
font = ImageFont.truetype(font_path, font_size)

# === ĐỌC LABEL VÀ VẼ ===
with open(label_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center, y_center, width, height = map(float, parts[1:])

    # Chuyển sang pixel
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    # Vẽ box và text bằng PIL
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
    label = f"{class_names[class_id]} ({class_id})"
    draw.text((x1, y1 - 20), label, font=font, fill="lime")

# === HIỂN THỊ ===
image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
cv2.imshow("Bounding Boxes (Tiếng Việt)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
