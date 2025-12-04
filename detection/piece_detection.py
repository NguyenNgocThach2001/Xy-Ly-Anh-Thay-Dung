import numpy as np
from ultralytics import YOLO

PAD_RATIO = 0.2  


def load_piece_model(model_path="colab_runs/weights/best.pt"):
    return YOLO(model_path)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _undo_rotation(cx, cy, W, H, rot):
    """Đưa toạ độ điểm (cx,cy) từ ảnh đã xoay về hệ 0° để mapping lưới.
    rot là góc đã xoay theo chiều kim đồng hồ trong board_detection: 0, 90, 180, 270.
    """
    rot = int(rot) % 360
    if rot == 0:
        return cx, cy
    elif rot == 90:
        # (x,y)_0 = (y, W-1-x)
        return cy, W - 1 - cx
    elif rot == 180:
        return W - 1 - cx, H - 1 - cy
    elif rot == 270:
        return H - 1 - cy, cx
    else:
        return cx, cy


def _map_row_with_river(y0, usable_len_rows, river_extra):
    """
    Ánh xạ toạ độ dọc theo trục Rows=10 (có 9 khoảng) với phần 'river_extra' chèn giữa
    sau 4 khoảng (giữa hàng 4 và 5). Trả về rowf thực (float).

    NGUYÊN TẮC (khớp overlay):
    - Kích thước mỗi ô (cell) KHÔNG đổi theo river_extra.
    - Tổng chiều cao thực tế = 9*cell + river_extra.
    - Biên dưới khe sông (river_end) ứng với HÀNG 5.0.
    """
    # cell phải giữ nguyên theo usable_len_rows/9 (không trừ river_extra)
    total_intervals = 9.0  # 10 hàng => 9 khoảng
    cell = max(1.0, float(usable_len_rows)) / total_intervals

    # Biên trước & sau sông
    river_start = 4.0 * cell
    river_end = river_start + float(river_extra)

    if y0 <= river_start:
        # Vùng trước sông: 0..4
        return y0 / cell
    elif y0 >= river_end:
        # Vùng sau sông: bắt đầu từ hàng 5.0
        return 5.0 + (y0 - river_end) / cell
    else:
        # Trong khe sông: kẹp về 4 hoặc 5 tuỳ nửa trên/dưới
        mid = (river_start + river_end) * 0.5
        return 4.0 if y0 < mid else 5.0


def detect_pieces_and_get_positions(model, board_img, grid_info, conf=0.8, iou=0.5, imgsz=640):
    """
    Trả về danh sách đúng format Pygame draw_pieces:
      [(label, (col,row)), ...]  với col ∈ [0..8], row ∈ [0..9]
    """
    if board_img is None or grid_info is None:
        return []

    H, W = board_img.shape[:2]
    class_names = model.model.names

    r = model.predict(source=board_img, conf=conf, iou=iou, imgsz=imgsz, verbose=False, device="cuda:0")[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy()

    # Lưới dựa theo padding/usable (khớp với ảnh warp hiển thị) + góc xoay
    pad_x = float(grid_info.get("pad_x", int(W * PAD_RATIO)))
    pad_y = float(grid_info.get("pad_y", int(H * PAD_RATIO)))
    usable_w = float(grid_info.get("usable_w", W - 2 * pad_x))
    usable_h = float(grid_info.get("usable_h", H - 2 * pad_y))
    rot = int(grid_info.get("rot", 0))

    swap_axes = bool(grid_info.get("swap_axes", False))
    river_extra = float(grid_info.get("river_extra", 0.0))
    scale_x = float(grid_info.get("scale_x", 1.0))
    scale_y = float(grid_info.get("scale_y", 1.0))

    # Offset dịch lưới theo X/Y (đơn vị pixel của ảnh warp)
    offset_x = float(grid_info.get("offset_x", 0.0))
    offset_y = float(grid_info.get("offset_y", 0.0))

    # áp scale vào usable (giữ schema giống overlay)
    eff_w = usable_w * max(0.2, min(5.0, scale_x))
    eff_h = usable_h * max(0.2, min(5.0, scale_y))

    # Chuyển bbox -> tâm -> un-rotate -> map lưới 9x10
    cand = []
    for b, sc, ci in zip(boxes, scores, clss):
        x1, y1, x2, y2 = b
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        cx, cy = _undo_rotation(cx, cy, W, H, rot)

        if not swap_axes:
            # Chuẩn: X -> 9 cột (8 khoảng), Y -> 10 hàng (9 khoảng, có river_extra)
            colf = ((cx - (pad_x + offset_x)) / max(1.0, eff_w)) * 8.0
            y0 = (cy - (pad_y + offset_y))
            rowf = _map_row_with_river(y0, eff_h, river_extra)
        else:
            # Hoán trục: Y -> 9 cột, X -> 10 hàng
            colf = ((cy - (pad_y + offset_y)) / max(1.0, eff_h)) * 8.0
            x0 = (cx - (pad_x + offset_x))
            rowf = _map_row_with_river(x0, eff_w, river_extra)

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
