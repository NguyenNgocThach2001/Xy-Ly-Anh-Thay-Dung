from ultralytics import YOLO
import cv2
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# === Load model
model = YOLO("runs/detect/xiangqi_fold5/weights/best.pt")
class_names = model.model.names  # lấy tên lớp từ model

# === Font hỗ trợ Unicode (tiếng Việt)
font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 20)

# === IoU tính toán giữa 2 box
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# === Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam.")
    exit()

# === Biến lưu thông tin frame trước
prev_boxes = []
prev_labels = []
last_infer_time = 0
infer_interval = 0.1  # giây
IOU_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    annotated = frame.copy()


    if now - last_infer_time > infer_interval:
        # === Inference
        results = model.predict(source=frame, conf=0.5)
        last_infer_time = now

        # === Lưu kết quả mới
        new_boxes = []
        new_labels = []

        for r in results:
            for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                b = box.cpu().numpy().astype(int)
                new_boxes.append(b)
                new_labels.append(int(cls_id.item()))

        # === So sánh IoU với box cũ
        merged_boxes = []
        merged_labels = []

        for new_box, new_label in zip(new_boxes, new_labels):
            keep = True
            for old_box, old_label in zip(prev_boxes, prev_labels):
                iou = compute_iou(new_box, old_box)
                if iou > IOU_THRESHOLD and new_label == old_label:
                    keep = False  # Đã có box tương tự rồi
                    break
            if keep:
                merged_boxes.append(new_box)
                merged_labels.append(new_label)

        prev_boxes = new_boxes
        prev_labels = new_labels

    # === Dùng PIL để hiển thị Unicode
    pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for box, cls_id in zip(prev_boxes, prev_labels):
        x1, y1, x2, y2 = box
        label = class_names.get(cls_id, str(cls_id))
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, y1 - 25), label, font=font, fill=(255, 255, 0))

    # === FPS
    fps = 1 / (time.time() - now + 1e-6)
    draw.text((10, 5), f"FPS: {fps:.1f}", font=font, fill=(255, 0, 0))

    # === Hiển thị
    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLOv8 + Stable Label", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
