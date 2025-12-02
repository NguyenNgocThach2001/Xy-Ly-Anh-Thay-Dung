import cv2
import numpy as np
from ultralytics import YOLO

PAD_RATIO = 0.2   # phải thống nhất với piece_detection

# =============================
# Helpers chung
# =============================
def extract_quad_from_mask(mask):
    mask = (mask.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            return quad
    return None

def order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # TL
    ordered[2] = pts[np.argmax(s)]  # BR
    ordered[1] = pts[np.argmin(diff)]  # TR
    ordered[3] = pts[np.argmax(diff)]  # BL
    return ordered

def align_board(frame, quad, output_size=(640, 640)):
    """
    Warp theo 4 điểm 'quad' VÀ KHÔNG tự xoay thêm.
    Góc xoay trả về luôn 0; việc xoay do UI (Rotate 90°) xử lý trước đó.
    """
    quad = order_quad_points(quad)
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    aligned = cv2.warpPerspective(frame, M, output_size)
    rot = 0  # KHÔNG auto-rotate, để UI quyết định
    return aligned, rot

def zoomout_after_align(image, pad_ratio=PAD_RATIO):
    h, w = image.shape[:2]
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    padded = cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    resized = cv2.resize(padded, (w, h))
    return resized, (pad_x, pad_y)

def load_board_model(model_path="runs/segment/yolov8_segment_board/weights/best.pt"):
    return YOLO(model_path)

# =============================
# Helper mới: warp frame bằng quad đã khóa
# =============================
def warp_with_quad(frame, quad, output_size=(640, 640)):
    """Warp 1 frame mới bằng 4 điểm quad đã có (theo toạ độ *gốc của frame hiện tại*).
    Trả về (board_img, grid_info)."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame phải là numpy.ndarray (H, W, 3) BGR")
    quad = np.array(quad, dtype=np.float32)
    aligned, rot = align_board(frame, quad, output_size=output_size)
    aligned_zoomout, (pad_x, pad_y) = zoomout_after_align(aligned, pad_ratio=PAD_RATIO)
    grid_info = {
        "pad_x": pad_x,
        "pad_y": pad_y,
        "usable_w": output_size[0] - 2 * pad_x,
        "usable_h": output_size[1] - 2 * pad_y,
        "rot": int(rot),
        "quad": quad.astype(float).tolist(),
        # ==== tham số manual-grid mặc định ====
        "swap_axes": False,   # False: X=9 cột, Y=10 hàng; True: hoán trục
        "river_extra": 0.0,   # px thêm cho khe sông
        "scale_x": 1.0,       # nhân khoảng cách cột
        "scale_y": 1.0,       # nhân khoảng cách hàng
        "offset_x": 0.0,      # dịch lưới theo X
        "offset_y": 0.0,      # dịch lưới theo Y
    }
    return aligned_zoomout, grid_info

# =============================
# YOLO detect
# =============================
def detect_board(model, frame):
    frame_resized = cv2.resize(frame, (640, 640))
    results = model.predict(source=frame_resized, conf=0.4, verbose=False)

    found_quad = None
    for r in results:
        if r.masks is not None:
            for mask in r.masks.data:
                quad = extract_quad_from_mask(mask)
                if quad is not None:
                    found_quad = quad
                    break

    if found_quad is not None:
        # Map quad (detected on resized 640x640) back to ORIGINAL frame coordinates
        H, W = frame.shape[:2]
        sx = W / 640.0
        sy = H / 640.0
        quad_orig = found_quad.astype(np.float32).copy()
        quad_orig[:, 0] *= sx
        quad_orig[:, 1] *= sy

        # Warp directly from ORIGINAL frame using quad in original coords
        aligned, rot = align_board(frame, quad_orig, output_size=(640, 640))
        aligned_zoomout, (pad_x, pad_y) = zoomout_after_align(aligned, pad_ratio=PAD_RATIO)
        grid_info = {
            "pad_x": pad_x,
            "pad_y": pad_y,
            "usable_w": 640 - 2 * pad_x,
            "usable_h": 640 - 2 * pad_y,
            "rot": int(rot),  # luôn 0 vì không auto-rotate
            "quad": quad_orig.astype(float).tolist(),
            # manual-grid mặc định
            "swap_axes": False,
            "river_extra": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
        return aligned_zoomout, True, grid_info

    return frame_resized, False, {"pad_x": 0, "pad_y": 0, "usable_w": 0, "usable_h": 0, "rot": 0}

# =============================
# GEOM mode (pipeline nhẹ)
# =============================

_last_warped = None
_last_quad = None
_last_mask = None
_stable_counter = 0
_STABLE_FRAMES = 10

def _order_points_rect(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

def _is_rectangle(pts, angle_thresh=15):
    def ang(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    pts = _order_points_rect(np.array(pts, dtype="float32"))
    angles = [ang(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return all(abs(a - 90) < angle_thresh for a in angles)

def _detect_geom_quad_and_warp(frame):
    global _last_warped, _last_quad, _last_mask, _stable_counter

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    roi = frame
    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            ordered = order_quad_points(approx.reshape(-1, 2).astype(np.float32))
            if _is_rectangle(ordered):
                # Warp từ frame gốc (không resize, không auto-rotate)
                aligned, rot = align_board(frame, ordered, output_size=(640, 640))
                aligned_zoomout, (pad_x, pad_y) = zoomout_after_align(aligned, pad_ratio=PAD_RATIO)

                # Giữ ổn định trong vài frame
                if _last_quad is not None:
                    dist = np.linalg.norm(_last_quad - ordered)
                    if dist < 10:
                        _stable_counter += 1
                    else:
                        _stable_counter = 0
                else:
                    _stable_counter = 0

                _last_warped = aligned_zoomout
                _last_quad = ordered
                _last_mask = edges

                if _stable_counter >= _STABLE_FRAMES:
                    grid_info = {
                        "pad_x": pad_x,
                        "pad_y": pad_y,
                        "usable_w": 640 - 2 * pad_x,
                        "usable_h": 640 - 2 * pad_y,
                        "rot": int(rot),  # luôn 0
                        "quad": _last_quad.astype(float).tolist(),
                        "swap_axes": False,
                        "river_extra": 0.0,
                        "scale_x": 1.0,
                        "scale_y": 1.0,
                        "offset_x": 0.0,
                        "offset_y": 0.0,
                    }
                    return aligned_zoomout, True, grid_info

    _last_warped = None
    _last_quad = None
    _last_mask = edges if roi.size > 0 else None
    frame_resized = cv2.resize(frame, (640, 640))
    return frame_resized, False, {"pad_x": 0, "pad_y": 0, "usable_w": 0, "usable_h": 0, "rot": 0}

# =============================
# Chọn mode
# =============================
def detect_board_with_mode(frame, mode="yolo", model=None, model_light=None, model_heavy=None):
    """
    mode:
      - 'yolo' : dùng YOLO duy nhất (ưu tiên model_light nếu có, nếu không dùng 'model')
      - 'geom' : pipeline hình học (nhẹ)
      - 'auto' : ưu tiên nhẹ trước: geom -> YOLO nhẹ -> YOLO nặng
    model:       (giữ tương thích cũ) coi như model_light nếu model_light=None
    model_light: YOLO nhẹ (ví dụ yolov8n)
    model_heavy: YOLO nặng (ví dụ yolov8m/l/x)
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame phải là numpy.ndarray (H, W, 3) BGR")

    # Chuẩn hóa tham số tương thích cũ
    if model_light is None and model is not None:
        model_light = model

    if mode == "yolo":
        if model_light is None:
            raise ValueError("Cần model hoặc model_light cho chế độ 'yolo'")
        return detect_board(model_light, frame)

    elif mode == "geom":
        return _detect_geom_quad_and_warp(frame)

    elif mode == "auto":
        # 1) Geom trước
        img, found, info = _detect_geom_quad_and_warp(frame)
        if found:
            return img, found, info

        # 2) YOLO nhẹ (nếu có)
        if model_light is not None:
            img, found, info = detect_board(model_light, frame)
            if found:
                return img, found, info

        # 3) YOLO nặng (nếu có)
        if model_heavy is not None:
            img, found, info = detect_board(model_heavy, frame)
            if found:
                return img, found, info

        # 4) Fallback cuối
        frame_resized = cv2.resize(frame, (640, 640))
        return frame_resized, False, {"pad_x": 0, "pad_y": 0, "usable_w": 0, "usable_h": 0, "rot": 0}

    else:
        raise ValueError("Mode không hợp lệ. Chọn 'yolo', 'geom' hoặc 'auto'.")
