# XỬ LÝ DỮ LIỆU CHO PROJECT NHẬN DẠNG CỜ TƯỚNG
## Data Preprocessing Pipeline

---

## SLIDE 1: GIỚI THIỆU PROJECT

### Project: Nhận dạng cờ tướng bằng Computer Vision

**Mục tiêu của project:**
- Phát hiện và nhận dạng các quân cờ trên bàn cờ tướng từ ảnh/video
- Phát hiện vị trí bàn cờ trong ảnh
- Xây dựng hệ thống tự động hóa việc ghi nhận nước đi trong cờ tướng

**Ứng dụng:**
- Ghi lại ván cờ tự động từ camera
- Phân tích và replay ván cờ
- Hỗ trợ học tập và nghiên cứu cờ tướng

**Công nghệ sử dụng:**
- YOLOv8 cho 2 task:
  - **Task 1**: Object Detection - Phát hiện quân cờ (14 classes)
  - **Task 2**: Instance Segmentation - Phát hiện bàn cờ (4 góc)

---

## SLIDE 2: YÊU CẦU VỀ BỘ DỮ LIỆU

### Để đạt được mục tiêu, cần bộ dữ liệu như thế nào?

**Yêu cầu cho Task 1 - Phát hiện quân cờ:**
- Ảnh bàn cờ với các quân cờ ở nhiều vị trí khác nhau
- Annotations cho 14 loại quân cờ:
  - Quân đen: Sĩ, Pháo, Xe, Tượng, Tướng, Mã, Tốt (7 loại)
  - Quân đỏ: Sĩ, Pháo, Xe, Tượng, Tướng, Mã, Tốt (7 loại)
- Đa dạng về góc chụp, ánh sáng, điều kiện môi trường

**Yêu cầu cho Task 2 - Phát hiện bàn cờ:**
- Ảnh bàn cờ từ nhiều góc độ khác nhau
- Annotations cho 4 góc của bàn cờ (polygon)
- Hỗ trợ perspective transformation để chuẩn hóa bàn cờ

**Chất lượng dữ liệu:**
- Độ phân giải đủ cao để nhận dạng rõ quân cờ
- Đa dạng về điều kiện chụp (ánh sáng, góc độ, khoảng cách)
- Số lượng ảnh đủ lớn để train model hiệu quả

---

## SLIDE 3: BỘ DỮ LIỆU NHÓM SỬ DỤNG

### 3 nguồn dữ liệu

**1. Dataset từ Roboflow - Chinese Chess Dataset 1**
- Nguồn: https://universe.roboflow.com/chinese-chess/chinese-zyx60
- Đặc điểm:
  - Đã có annotations sẵn
  - Format: JSON (Roboflow format)
  - Bao gồm cả quân cờ và bàn cờ

**2. Dataset từ Roboflow - Chinese Chess Dataset 2**
- Nguồn: https://universe.roboflow.com/viktor-ng/chinese-chess-rtpmq
- Đặc điểm:
  - Đã có annotations sẵn
  - Format: JSON (Roboflow format)
  - Bổ sung thêm dữ liệu đa dạng

**3. Dataset tự tạo**
- Tự chụp bàn cờ và quân cờ ở nhiều góc độ khác nhau
- Đặc điểm:
  - Chưa có label
  - Chưa được xử lý
  - Cần annotation thủ công hoặc semi-automatic

**Tổng hợp:**
- 2 bộ có sẵn từ Roboflow → Format JSON, đã có label
- 1 bộ tự chụp → Chưa có label, cần xử lý

---

## SLIDE 4: THÁCH THỨC VÀ GIẢI PHÁP

### Thách thức khi làm việc với 3 nguồn dữ liệu

**Thách thức 1: Format khác nhau**
- 2 dataset Roboflow: Format JSON (Roboflow annotation format)
- Dataset tự chụp: Chưa có format, chưa có label
- YOLOv8 yêu cầu: Format YOLO (.txt files)

**Thách thức 2: ID class không thống nhất**
- Dataset 1 có thể dùng ID: 0, 1, 2, ...
- Dataset 2 có thể dùng ID: 0, 1, 2, ... (nhưng mapping khác)
- Cần chuẩn hóa về 1 bộ ID thống nhất (0-13)

**Thách thức 3: Dataset tự chụp chưa có label**
- Cần annotation thủ công hoặc sử dụng tool
- Cần chuyển đổi sang format YOLO

**Thách thức 4: Gộp nhiều dataset**
- Trùng tên file giữa các dataset
- Cần đảm bảo không mất dữ liệu
- Cần tạo file config YAML thống nhất

---

## SLIDE 5: QUY TRÌNH XỬ LÝ DỮ LIỆU - TỔNG QUAN

### Pipeline xử lý dữ liệu

```
┌─────────────────┐
│ 3 Nguồn Data    │
│ - Roboflow 1    │
│ - Roboflow 2    │
│ - Tự chụp       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bước 1: Chuyển  │
│ JSON → YOLO     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bước 2: Chuẩn   │
│ hóa ID          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bước 3: Gộp     │
│ Dataset         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bước 4: Chia    │
│ Train/Val/Test  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dataset sẵn     │
│ sàng cho train  │
└─────────────────┘
```

---

## SLIDE 6: BƯỚC 1 - CHUYỂN ĐỔI JSON SANG YOLO FORMAT

### Script: `json2detection.py` và `json2seg.py`

**Input:**
- `rawdata/images/` - Ảnh từ Roboflow
- `rawdata/labels/*.json` - Annotations dạng JSON

**Xử lý:**
1. **Cho Task Detection (quân cờ):**
   - Đọc JSON annotations
   - Trích xuất bounding box từ `rectMask` hoặc `content` (polygon)
   - Chuyển đổi sang YOLO format: `class_id x_center y_center width height`
   - Normalize tọa độ về [0, 1]
   - Chuẩn hóa tên class (ví dụ: "guard" → "advisor")

2. **Cho Task Segmentation (bàn cờ):**
   - Đọc JSON annotations
   - Trích xuất polygon 4 góc bàn cờ
   - Chuyển đổi sang YOLO segmentation format: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
   - Normalize tọa độ về [0, 1]

**Output:**
- `processed_data_detection/` - Dataset cho detection
- `seg_data/` - Dataset cho segmentation

**Code xử lý:**
```python
# Trích xuất bbox từ JSON
def obj_to_bbox(obj):
    if "rectMask" in obj:
        return rectmask_to_bbox(obj["rectMask"])
    elif "content" in obj:
        return polygon_to_bbox(obj["content"])
    
# Chuyển sang YOLO format
x_center = (xmin + xmax) / 2.0 / img_w
y_center = (ymin + ymax) / 2.0 / img_h
box_w = (xmax - xmin) / img_w
box_h = (ymax - ymin) / img_h
```

---

## SLIDE 7: BƯỚC 2 - CHUẨN HÓA ID CLASS

### Script: `change_dts_id.py`

**Vấn đề:**
- Mỗi dataset có ID class riêng (có thể trùng hoặc khác)
- Cần chuẩn hóa trước khi gộp

**Giải pháp:**
1. Đọc file `data.yaml` của từng dataset
2. Tạo mapping từ ID cũ → ID tạm (100-113)
   - black-advisor → 100
   - red-advisor → 101
   - black-cannon → 104
   - ... (14 classes)
3. Cập nhật tất cả file label `.txt` trong train/val/test
4. Tạo file `remapped.yaml` mới

**Code xử lý:**
```python
# Mapping ID
LABEL_MAP = {
    'black-advisor': 100,
    'red-advisor': 101,
    # ... 14 classes
}

# Cập nhật label files
for line in label_file:
    old_id = int(parts[0])
    new_id = oldid_to_newid[old_id]
    parts[0] = str(new_id)
```

**Output:**
- `Dataset_Detection/piece/remapped_dts1/` - Dataset 1 đã remap
- `Dataset_Detection/piece/remapped_dts2/` - Dataset 2 đã remap

---

## SLIDE 8: BƯỚC 3 - GỘP DATASET

### Script: `merge2dts.py`

**Mục tiêu:**
- Gộp 2 dataset đã remap (ID 100-113)
- Chuyển ID về range chuẩn 0-13
- Xử lý trùng tên file

**Quy trình:**
1. Đọc 2 dataset đã remap
2. Copy tất cả ảnh và labels
3. Xử lý trùng tên: đổi tên file (thêm suffix `_dts1`, `_dts2`)
4. Remap ID từ 100-113 → 0-13
5. Tạo file `data.yaml` thống nhất

**Mapping ID cuối cùng:**
```python
ID_MAP = {
    100: 0,   # black-advisor
    104: 1,   # black-cannon
    106: 2,   # black-chariot
    102: 3,   # black-elephant
    108: 4,   # black-general
    110: 5,   # black-horse
    112: 6,   # black-soldier
    101: 7,   # red-advisor
    105: 8,   # red-cannon
    107: 9,   # red-chariot
    103: 10,  # red-elephant
    109: 11,  # red-general
    111: 12,  # red-horse
    113: 13   # red-soldier
}
```

**Output:**
- `Dataset_Detection/piece/Dataset_combined/` - Dataset đã gộp và chuẩn hóa

---

## SLIDE 9: BƯỚC 4 - CHIA DỮ LIỆU (TÙY CHỌN)

### Script: `data_prepare.py` - K-fold Cross-validation

**Mục đích:**
- Chia dataset thành K folds để cross-validation
- Đảm bảo đánh giá model khách quan

**Quy trình:**
1. Lấy tất cả ảnh từ `dataset/train/`
2. Shuffle với random seed cố định
3. Chia thành K folds bằng `sklearn.model_selection.KFold`
4. Tạo thư mục cho mỗi fold:
   - `images/train_fold1/`, `val_fold1/`
   - `labels/train_fold1/`, `val_fold1/`
5. Tạo file `config_fold1.yaml`, `config_fold2.yaml`, ...

**Cấu trúc output:**
```
dataset/
├── images/
│   ├── train_fold1/
│   ├── val_fold1/
│   ├── train_fold2/
│   └── ...
├── labels/
│   ├── train_fold1/
│   ├── val_fold1/
│   └── ...
└── config_fold1.yaml, config_fold2.yaml, ...
```

**Lợi ích:**
- Đánh giá model trên nhiều tập dữ liệu khác nhau
- Tránh overfitting
- Tận dụng tối đa dữ liệu

---

## SLIDE 10: THÔNG TIN TỔNG QUÁT BỘ DỮ LIỆU SAU XỬ LÝ

### Dataset cuối cùng

**Task 1 - Piece Detection (Phát hiện quân cờ):**
- **Tổng số ảnh:** [Số liệu thực tế]
- **Số classes:** 14
  - Quân đen (0-6): Sĩ, Pháo, Xe, Tượng, Tướng, Mã, Tốt
  - Quân đỏ (7-13): Sĩ, Pháo, Xe, Tượng, Tướng, Mã, Tốt
- **Chia dữ liệu:**
  - Train: [X] ảnh
  - Validation: [Y] ảnh
  - Test: [Z] ảnh
- **Format:** YOLO detection format
  - Mỗi file `.txt`: `class_id x_center y_center width height`

**Task 2 - Board Segmentation (Phát hiện bàn cờ):**
- **Tổng số ảnh:** [Số liệu thực tế]
- **Số classes:** 1 (board)
- **Chia dữ liệu:**
  - Train: [X] ảnh
  - Validation: [Y] ảnh
  - Test: [Z] ảnh
- **Format:** YOLO segmentation format
  - Mỗi file `.txt`: `class_id x1 y1 x2 y2 x3 y3 x4 y4`

**File cấu hình:**
- `Dataset_Detection/piece/Dataset_combined/data.yaml` - Config cho piece detection
- `seg_data/data.yaml` - Config cho board segmentation

---

## SLIDE 11: CHIA DỮ LIỆU CHO TRAINING

### Chia dữ liệu cho 2 task

**Task 1 - Piece Detection:**
```
Dataset_Detection/piece/Dataset_combined/
├── data.yaml
├── train/
│   ├── images/      # [X] ảnh
│   └── labels/      # [X] file .txt
├── val/
│   ├── images/      # [Y] ảnh
│   └── labels/      # [Y] file .txt
└── test/
    ├── images/      # [Z] ảnh
    └── labels/      # [Z] file .txt
```

**Task 2 - Board Segmentation:**
```
seg_data/
├── data.yaml
├── train/
│   ├── images/      # [X] ảnh
│   └── labels/      # [X] file .txt (polygon)
├── val/
│   ├── images/      # [Y] ảnh
│   └── labels/      # [Y] file .txt
└── test/
    ├── images/      # [Z] ảnh
    └── labels/      # [Z] file .txt
```

**Tỷ lệ chia (ví dụ):**
- Train: 70%
- Validation: 15%
- Test: 15%

**Sử dụng với YOLOv8:**
```python
# Train piece detection
model = YOLO('yolov8n.pt')
model.train(
    data='Dataset_Detection/piece/Dataset_combined/data.yaml',
    epochs=100,
    imgsz=640
)

# Train board segmentation
model = YOLO('yolov8n-seg.pt')
model.train(
    data='seg_data/data.yaml',
    epochs=100,
    imgsz=640
)
```

---

## SLIDE 12: TỔNG KẾT

### Quy trình xử lý dữ liệu hoàn chỉnh

**Đã thực hiện:**
1. ✅ Chuyển đổi 2 dataset Roboflow từ JSON → YOLO format
2. ✅ Xử lý dataset tự chụp (annotation và chuyển đổi)
3. ✅ Chuẩn hóa ID class cho tất cả dataset
4. ✅ Gộp 3 nguồn dữ liệu thành 1 dataset thống nhất
5. ✅ Chia dữ liệu cho 2 task: detection và segmentation
6. ✅ Tạo file config YAML cho YOLOv8

**Kết quả:**
- Dataset sẵn sàng cho training YOLOv8
- 2 bộ dataset riêng biệt cho 2 task
- Format chuẩn, ID thống nhất
- Có thể sử dụng K-fold cross-validation

**Công cụ:**
- 5 Python scripts tự động hóa toàn bộ quy trình
- Code có comment rõ ràng, dễ bảo trì
- Xử lý lỗi và edge cases

---

## SLIDE 13: CẤU TRÚC CLASS NAMES

### 14 Classes cho Piece Detection

**Quân đen (ID 0-6):**
- 0: black-advisor (Sĩ đen)
- 1: black-cannon (Pháo đen)
- 2: black-chariot (Xe đen)
- 3: black-elephant (Tượng đen)
- 4: black-general (Tướng đen)
- 5: black-horse (Mã đen)
- 6: black-soldier (Tốt đen)

**Quân đỏ (ID 7-13):**
- 7: red-advisor (Sĩ đỏ)
- 8: red-cannon (Pháo đỏ)
- 9: red-chariot (Xe đỏ)
- 10: red-elephant (Tượng đỏ)
- 11: red-general (Tướng đỏ)
- 12: red-horse (Mã đỏ)
- 13: red-soldier (Tốt đỏ)

**Board Segmentation:**
- 0: board (bàn cờ) - 4 điểm polygon

---

## SLIDE 14: DEMO CODE STRUCTURE

### Cấu trúc code xử lý

```
datapreprocessing/
├── json2detection.py    # Chuyển JSON → YOLO detection
├── json2seg.py          # Chuyển JSON → YOLO segmentation
├── change_dts_id.py     # Chuẩn hóa ID class
├── merge2dts.py         # Gộp dataset
├── data_prepare.py      # K-fold splits
└── README.md            # Hướng dẫn sử dụng
```

**Đặc điểm code:**
- ✅ Comment tiếng Việt rõ ràng
- ✅ Xử lý lỗi đầy đủ
- ✅ Type hints cho các hàm
- ✅ Tự động tạo thư mục output
- ✅ Logging và thông báo chi tiết

**Ví dụ sử dụng:**
```bash
# Bước 1: Chuyển đổi JSON
python datapreprocessing/json2detection.py
python datapreprocessing/json2seg.py

# Bước 2: Chuẩn hóa ID (cho mỗi dataset)
python datapreprocessing/change_dts_id.py

# Bước 3: Gộp dataset
python datapreprocessing/merge2dts.py

# Bước 4: K-fold (tùy chọn)
python datapreprocessing/data_prepare.py
```

---

## SLIDE 15: KẾT LUẬN

### Thành công xử lý dữ liệu từ 3 nguồn khác nhau

**Đạt được:**
- ✅ Tự động hóa quy trình xử lý dữ liệu
- ✅ Chuẩn hóa format và ID class
- ✅ Gộp thành công 3 nguồn dữ liệu
- ✅ Dataset sẵn sàng cho training

**Sẵn sàng cho bước tiếp theo:**
- Training YOLOv8 cho piece detection
- Training YOLOv8 cho board segmentation
- Đánh giá và tối ưu model

**Lợi ích:**
- Tiết kiệm thời gian xử lý thủ công
- Đảm bảo tính nhất quán của dữ liệu
- Dễ dàng mở rộng và bảo trì

---

## CẢM ƠN!

### Questions?

