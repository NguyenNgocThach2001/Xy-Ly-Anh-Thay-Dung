from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

#MODEL_PATH = "runs/segment/yolov8_segment_board/weights/best.pt"   
MODEL_PATH = "model_ban_co_mau.pt"  
model = YOLO(MODEL_PATH)

image_files = [
    "ban_co_mau.jpg",
    "ban_co_mau_1.jpg", 
    "ban_co_mau_2.jpg",
    "ban_co_mau_3.jpg", 
    "ban_co_mau_4.jpg",
    "ban_co_mau_5.jpg",
    "ban_co_mau_6.jpg",
    "ban_co_mau_7.jpg",
    "ban_co_mau_8.jpg",
]

num_images = len(image_files)
fig, axes = plt.subplots(1, num_images, figsize=(8 * num_images, 8))

if num_images == 1:
    axes = [axes]


for idx, image_path in enumerate(image_files):
    if not os.path.exists(image_path):
        print(f"Cảnh báo: Không tìm thấy file {image_path}")
        continue
    
    img = cv2.imread(image_path)
    
    results = model.predict(
        source=img,       
        conf=0.9,
        verbose=False
    )
    
    # Lấy ảnh đã vẽ mask/bbox
    result = results[0]
    annotated = result.plot()   # BGR (OpenCV format)
    # Chuyển sang RGB
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Hiển thị
    axes[idx].imshow(annotated_rgb)
    axes[idx].axis("off")
    axes[idx].set_title(f"{image_path}")

plt.tight_layout()
plt.show()