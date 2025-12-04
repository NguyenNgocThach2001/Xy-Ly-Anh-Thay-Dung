from ultralytics import YOLO
import cv2
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# ==== CẤU HÌNH ====
CHECKPOINT_PATH = "runs/detect/model_on_dataset_combined_500_epoch/weights/best.pt"
CAMERA_SOURCE = "http://192.168.1.97:4747/video"  # 0 = webcam, hoặc dùng link IP cam như "http://192.168.1.97:4747/video"
CONFIDENCE_THRESHOLD = 0.7
FONT_PATH = "C:/Windows/Fonts/arial.ttf"

# ==== LOAD MODEL ====
model = YOLO(CHECKPOINT_PATH)
class_names = model.model.names
font = ImageFont.truetype(FONT_PATH, 20)

# ==== LETTERBOX ====
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / w, new_shape[1] / h)
    resized_w, resized_h = int(w * r), int(h * r)
    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[0] - resized_w, new_shape[1] - resized_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)

# ==== MỞ CAMERA ====
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print("Không mở được camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    h, w = frame.shape[:2]
    original = frame.copy()

    # Letterbox input ảnh
    padded, scale, (pad_x, pad_y) = letterbox(frame, (640, 640))

    # === INFERENCE ===
    results = model.predict(source=padded, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # PIL để hỗ trợ font tiếng Việt
    pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for r in results:
        for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
            b = box.cpu().numpy()

            # Scale bbox ngược về ảnh gốc
            b[0::2] -= pad_x
            b[1::2] -= pad_y
            b /= scale
            b = np.clip(b, [0, 0, 0, 0], [w, h, w, h])
            x1, y1, x2, y2 = b.astype(int)

            label = class_names.get(int(cls_id.item()), str(int(cls_id.item())))
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            draw.text((x1, y1 - 25), label, font=font, fill=(255, 255, 0))

    fps = 1 / (time.time() - start + 1e-6)
    draw.text((10, 5), f"FPS: {fps:.1f}", font=font, fill=(255, 0, 0))

    # Show kết quả
    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLOv8 Inference", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
