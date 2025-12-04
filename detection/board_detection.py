import cv2
import numpy as np
from ultralytics import YOLO

PAD_RATIO = 0.05

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

def zoomout_quad(quad, ratio=PAD_RATIO):
    """
    Nới rộng 4 góc quad ra thêm một khoảng so với tâm quad.
    - quad: (4, 2) np.float32
    - ratio: tỉ lệ nới rộng (0.2 tức là nở ra 20% mỗi chiều)
    """
    quad = np.array(quad, dtype=np.float32)
    center = quad.mean(axis=0)
    zoomed = center + (quad - center) * (1 + ratio)
    return zoomed

def align_board(frame, quad, output_size=(640, 640), pad_ratio=PAD_RATIO):
    """
    Warp qua 4 điểm 'quad'
    NẾU pad_ratio > 0, sẽ zoom out quad rồi mới align, không tạo viền bằng copyMakeBorder nữa!
    Góc xoay trả về luôn 0; việc xoay do UI (Rotate 90°) xử lý trước đó.
    """
    quad = order_quad_points(quad)
    if pad_ratio != 0.0:
        quad = zoomout_quad(quad, pad_ratio)
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    aligned = cv2.warpPerspective(frame, M, output_size)
    rot = 0
    pad_x = 0
    pad_y = 0
    usable_w = output_size[0]
    usable_h = output_size[1]
    if pad_ratio != 0.0:
        pad_x = int(output_size[0] * pad_ratio / (1 + pad_ratio * 2))
        pad_y = int(output_size[1] * pad_ratio / (1 + pad_ratio * 2))
        usable_w = output_size[0] - 2 * pad_x
        usable_h = output_size[1] - 2 * pad_y
    return aligned, rot, pad_x, pad_y, usable_w, usable_h

def load_board_model(model_path="runs/segment/yolov8_segment_board/weights/best.pt"):
    return YOLO(model_path)

# warp frame bằng quad đã khóa
def warp_with_quad(frame, quad, output_size=(640, 640), pad_ratio=PAD_RATIO):
    """Trả về (board_img, grid_info)
    Warp + zoom out quad để lấy ngay viền ngoài"""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame phải là numpy.ndarray (H, W, 3) BGR")
    quad = np.array(quad, dtype=np.float32)
    aligned, rot, pad_x, pad_y, usable_w, usable_h = align_board(frame, quad, output_size=output_size, pad_ratio=pad_ratio)
    grid_info = {
        "pad_x": pad_x,
        "pad_y": pad_y,
        "usable_w": usable_w,
        "usable_h": usable_h,
        "rot": int(rot),
        "quad": quad.astype(float).tolist(),
        # ==== tham số manual-grid mặc định ====
        "swap_axes": False,
        "river_extra": 0.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
    }
    return aligned, grid_info

# YOLO detect
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

        # Warp & zoomout trực tiếp từ ORIGINAL frame using quad in original coords
        aligned, rot, pad_x, pad_y, usable_w, usable_h = align_board(
            frame, quad_orig, output_size=(640, 640), pad_ratio=PAD_RATIO
        )
        grid_info = {
            "pad_x": pad_x,
            "pad_y": pad_y,
            "usable_w": usable_w,
            "usable_h": usable_h,
            "rot": int(rot),
            "quad": quad_orig.astype(float).tolist(),
            # manual-grid mặc định
            "swap_axes": False,
            "river_extra": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
        return aligned, True, grid_info

    return frame_resized, False, {"pad_x": 0, "pad_y": 0, "usable_w": 0, "usable_h": 0, "rot": 0}

# GEOM mode

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
                # Warp + zoomout bằng quad đã mở rộng
                aligned, rot, pad_x, pad_y, usable_w, usable_h = align_board(
                    frame, ordered, output_size=(640, 640), pad_ratio=PAD_RATIO
                )
                # Giữ ổn định trong vài frame
                if _last_quad is not None:
                    dist = np.linalg.norm(_last_quad - ordered)
                    if dist < 10:
                        _stable_counter += 1
                    else:
                        _stable_counter = 0
                else:
                    _stable_counter = 0

                _last_warped = aligned
                _last_quad = ordered
                _last_mask = edges

                if _stable_counter >= _STABLE_FRAMES:
                    grid_info = {
                        "pad_x": pad_x,
                        "pad_y": pad_y,
                        "usable_w": usable_w,
                        "usable_h": usable_h,
                        "rot": int(rot),
                        "quad": _last_quad.astype(float).tolist(),
                        "swap_axes": False,
                        "river_extra": 0.0,
                        "scale_x": 1.0,
                        "scale_y": 1.0,
                        "offset_x": 0.0,
                        "offset_y": 0.0,
                    }
                    return aligned, True, grid_info

    _last_warped = None
    _last_quad = None
    _last_mask = edges if roi.size > 0 else None
    frame_resized = cv2.resize(frame, (640, 640))
    return frame_resized, False, {"pad_x": 0, "pad_y": 0, "usable_w": 0, "usable_h": 0, "rot": 0}

# Chọn mode
def detect_board_with_mode(frame, mode="yolo", model=None, model_light=None, model_heavy=None):
    if model_light is None and model is not None:
        model_light = model

    if mode == "yolo":
        if model_light is None:
            raise ValueError("Cần model hoặc model_light cho chế độ 'yolo'")
        return detect_board(model_light, frame)

    elif mode == "geom":
        return _detect_geom_quad_and_warp(frame)

    elif mode == "auto":
        # 1) Geom
        img, found, info = _detect_geom_quad_and_warp(frame)
        if found:
            return img, found, info

        # 2) YOLO nhẹ
        if model_light is not None:
            img, found, info = detect_board(model_light, frame)
            if found:
                return img, found, info

        # 3) YOLO nặng
        if model_heavy is not None:
            img, found, info = detect_board(model_heavy, frame)
            if found:
                return img, found, info

        # 4) Fallback
        frame_resized = cv2.resize(frame, (640, 640))
        return frame_resized, False, {"pad_x": 0, "pad_y": 0, "usable_w": 0, "usable_h": 0, "rot": 0}