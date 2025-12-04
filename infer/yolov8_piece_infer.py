from ultralytics import YOLO
import cv2
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# === Load model
model = YOLO("runs/detect/model_on_dataset1_200_epoch/weights/best.pt")
class_names = model.model.names

# === Font hỗ trợ tiếng Việt
font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 20)

# === Letterbox ảnh vào 640x640
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    resized_w, resized_h = int(w * r), int(h * r)
    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_w - resized_w, new_h - resized_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)

# === Kết nối webcam
cap = cv2.VideoCapture("http://192.168.1.97:4747/video")
if not cap.isOpened():
    print("❌ Không mở được webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    h, w = frame.shape[:2]
    original = frame.copy()

    # === Letterbox
    padded, scale, (pad_x, pad_y) = letterbox(frame, (640, 640))

    # === Inference
    results = model.predict(source=padded, conf=0.8, verbose=False)

    # === Vẽ kết quả
    pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for r in results:
        for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
            b = box.cpu().numpy()

            # Scale ngược bbox về ảnh gốc
            b[0::2] -= pad_x
            b[1::2] -= pad_y
            b /= scale
            b = np.clip(b, [0, 0, 0, 0], [w, h, w, h])
            b = b.astype(int)

            x1, y1, x2, y2 = b
            label = class_names.get(int(cls_id.item()), str(int(cls_id.item())))
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            draw.text((x1, y1 - 25), label, font=font, fill=(255, 255, 0))

    fps = 1 / (time.time() - start + 1e-6)
    draw.text((10, 5), f"FPS: {fps:.1f}", font=font, fill=(255, 0, 0))

    # === Show
    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLOv8 (No IoU)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
