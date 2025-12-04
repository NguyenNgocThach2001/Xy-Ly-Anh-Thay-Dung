# Hướng dẫn Data Preprocessing

Thư mục này chứa các script để xử lý dữ liệu từ 2 dataset Roboflow:
- [Chinese Chess Dataset 1](https://universe.roboflow.com/chinese-chess/chinese-zyx60)
- [Chinese Chess Dataset 2](https://universe.roboflow.com/viktor-ng/chinese-chess-rtpmq)

## Quy trình xử lý dữ liệu

### Bước 1: Tải dataset từ Roboflow
1. Tải 2 dataset từ Roboflow ở định dạng YOLO
2. Giải nén và đặt vào cấu trúc thư mục như sau:
   ```
   rawdata/
       images/          # Ảnh từ cả 2 dataset
       labels/          # File JSON annotations từ Roboflow
   ```

### Bước 2: Chuyển đổi JSON sang YOLO format

#### 2.1. Chuyển đổi cho piece detection (quân cờ)
```bash
python datapreprocessing/json2detection.py
```
- **Input**: `rawdata/images/` và `rawdata/labels/*.json`
- **Output**: `processed_data_detection/images/` và `processed_data_detection/labels/*.txt`
- **Yêu cầu**: File `Dataset_Detection/piece/Dataset_combined/data.yaml` phải tồn tại (hoặc tạo trước)

#### 2.2. Chuyển đổi cho board detection (bàn cờ)
```bash
python datapreprocessing/json2seg.py
```
- **Input**: `rawdata/images/` và `rawdata/labels/*.json`
- **Output**: `seg_data/images/` và `seg_data/labels/*.txt`
- **Mục đích**: Tạo dataset segmentation cho việc phát hiện 4 góc bàn cờ

### Bước 3: Chuẩn hóa ID cho từng dataset

Chạy script này cho mỗi dataset riêng biệt:

```bash
python datapreprocessing/change_dts_id.py
```

**Lưu ý**: Cần chỉnh sửa biến `ORIGINAL_YAML_PATH` trong file để trỏ đến dataset cụ thể.

- **Input**: Dataset gốc (ví dụ: `Dataset_Detection/piece/Dataset4_270image/`)
- **Output**: Dataset đã remap (ví dụ: `Dataset_Detection/piece/Dataset4_270image/remapped_dts/`)
- **Chức năng**: Chuyển đổi ID từ range gốc sang range 100-113 để chuẩn bị merge

### Bước 4: Gộp 2 dataset

```bash
python datapreprocessing/merge2dts.py
```

- **Input**: 
  - `Dataset_Detection/piece/remapped_dts1/` (dataset 1 đã remap)
  - `Dataset_Detection/piece/remapped_dts2/` (dataset 2 đã remap)
- **Output**: `Dataset_Detection/piece/Dataset_combined/`
- **Chức năng**: 
  - Gộp tất cả ảnh và labels từ 2 dataset
  - Chuyển đổi ID từ 100-113 về 0-13 (ID chuẩn)
  - Tạo file `data.yaml` mới

### Bước 5: Tạo K-fold splits (tùy chọn)

```bash
python datapreprocessing/data_prepare.py
```

- **Input**: `dataset/train/images/` và `dataset/train/labels/`
- **Output**: 
  - `dataset/images/train_fold1/`, `val_fold1/`, ...
  - `dataset/labels/train_fold1/`, `val_fold1/`, ...
  - `dataset/config_fold1.yaml`, `config_fold2.yaml`, ...
- **Chức năng**: Chia dataset thành K folds để cross-validation

## Cấu trúc class names

Dataset sử dụng 14 classes cho quân cờ:

**Quân đen (0-6):**
- 0: black-advisor (Sĩ đen)
- 1: black-cannon (Pháo đen)
- 2: black-chariot (Xe đen)
- 3: black-elephant (Tượng đen)
- 4: black-general (Tướng đen)
- 5: black-horse (Mã đen)
- 6: black-soldier (Tốt đen)

**Quân đỏ (7-13):**
- 7: red-advisor (Sĩ đỏ)
- 8: red-cannon (Pháo đỏ)
- 9: red-chariot (Xe đỏ)
- 10: red-elephant (Tượng đỏ)
- 11: red-general (Tướng đỏ)
- 12: red-horse (Mã đỏ)
- 13: red-soldier (Tốt đỏ)

## Cấu hình

Trước khi chạy các script, cần chỉnh sửa các biến cấu hình trong từng file:

1. **json2detection.py**: 
   - `ROOT`: Đường dẫn gốc của project
   - `DATA_YAML`: Đường dẫn đến file data.yaml

2. **json2seg.py**:
   - `ROOT`: Đường dẫn gốc của project

3. **change_dts_id.py**:
   - `ORIGINAL_YAML_PATH`: Đường dẫn đến file YAML của dataset cần remap

4. **merge2dts.py**:
   - `REMAP_DIRS`: Danh sách thư mục dataset đã remap
   - `OUTPUT_DIR`: Thư mục output cho dataset đã gộp

5. **data_prepare.py**:
   - `IMAGE_DIR`: Thư mục chứa ảnh training
   - `LABEL_DIR`: Thư mục chứa labels training
   - `NUM_FOLDS`: Số lượng folds (mặc định: 5)
   - `CLASS_NAMES`: Danh sách tên class

## Lưu ý

- Tất cả các script đều có comment tiếng Việt rõ ràng
- Các script tự động tạo thư mục output nếu chưa tồn tại
- Script sẽ bỏ qua các file không hợp lệ và in cảnh báo
- Đảm bảo có đủ dung lượng ổ cứng vì script sẽ copy toàn bộ dataset

