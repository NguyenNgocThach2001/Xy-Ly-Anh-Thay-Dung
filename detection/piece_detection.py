import numpy as np
from ultralytics import YOLO

PAD_RATIO = 0.10  # phải khớp với board_detection

def load_piece_model(model_path="runs/detect/model_on_dataset_combined_500_epoch_23/weights/best.pt"):
    return YOLO(model_path)

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def detect_pieces_and_get_positions(model, board_img, grid_info, conf=0.8, iou=0.5, imgsz=640):
    """
    Trả về danh sách đúng format Pygame draw_pieces:
      [(label, (col,row)), ...]  với col ∈ [0..8], row ∈ [0..9]
    """
    H, W = board_img.shape[:2]
    class_names = model.model.names

    r = model.predict(source=board_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    clss   = r.boxes.cls.cpu().numpy()

    # Lưới dựa theo padding/usable (khớp với ảnh warp hiển thị)
    pad_x = grid_info.get("pad_x", int(W * PAD_RATIO))
    pad_y = grid_info.get("pad_y", int(H * PAD_RATIO))
    usable_w = grid_info.get("usable_w", W - 2 * pad_x)
    usable_h = grid_info.get("usable_h", H - 2 * pad_y)

    # Chuyển bbox -> tâm -> nút lưới
    cand = []
    for b, sc, ci in zip(boxes, scores, clss):
        x1, y1, x2, y2 = b
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        colf = (cx - pad_x) / max(1, usable_w) * 8.0
        rowf = (cy - pad_y) / max(1, usable_h) * 9.0
        col = _clamp(int(round(colf)), 0, 8)
        row = _clamp(int(round(rowf)), 0, 9)
        label = class_names.get(int(ci), str(int(ci)))
        cand.append((label, (col, row), float(sc)))

    # Giải quyết trùng ô: giữ detection có confidence cao nhất
    best = {}
    for name, pos, score in cand:
        if pos not in best or score > best[pos][1]:
            best[pos] = (name, score)

    pieces_for_pygame = [(name, pos) for pos, (name, _) in best.items()]
    return pieces_for_pygame
