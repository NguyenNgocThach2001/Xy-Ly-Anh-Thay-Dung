import cv2
import numpy as np
from ultralytics import YOLO

# =============================
# Tham số
# =============================
EXPAND_RATIO = 1.06   # nới rộng polygon trước khi warp (>=1.0). Ví dụ 1.05 ~ 5%
PAD_RATIO    = 0.0    # pad viền đen sau khi align (rồi resize về kích cỡ cũ). 0.0 = không pad

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
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2)
    return None

def order_quad_points(pts):
    center = np.mean(pts, axis=0)
    def angle(pt): return np.arctan2(pt[1] - center[1], pt[0] - center[0])
    pts_sorted = sorted(pts, key=angle)
    pts_sorted = np.array(pts_sorted, dtype=np.float32)
    top_left_idx = np.argmin(np.sum(pts_sorted, axis=1))
    pts_sorted = np.roll(pts_sorted, -top_left_idx, axis=0)
    return pts_sorted

def expand_quad(quad, ratio=1.0):
    """Nới rộng hoặc thu hẹp polygon quanh tâm. ratio > 1.0 = nới rộng."""
    if ratio == 1.0:
        return quad.astype(np.float32)
    quad = quad.astype(np.float32)
    center = np.mean(quad, axis=0, keepdims=True)
    return center + (quad - center) * float(ratio)

def _auto_rotate_long_side_vertical(aligned, quad_ordered):
    """Đảm bảo cạnh dài (10 ô) nằm theo chiều dọc.
    Trả về (img_rotated, rot_deg_clockwise) với rot ∈ {0,90}.
    """
    tl, tr, br, bl = quad_ordered
    top_len = np.linalg.norm(tr - tl)
    left_len = np.linalg.norm(bl - tl)
    if top_len > left_len:
        return cv2.rotate(aligned, cv2.ROTATE_90_CLOCKWISE), 90
    return aligned, 0

def align_board(frame, quad, output_size=(640, 640), expand_ratio=1.0):
    """Warp bàn cờ về khung vuông, có nới rộng quad trước khi warp."""
    quad = order_quad_points(quad)
    quad_expanded = expand_quad(quad, expand_ratio)

    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad_expanded, dst)
    aligned = cv2.warpPerspective(frame, M, output_size)
    aligned, rot = _auto_rotate_long_side_vertical(aligned, quad_expanded)
    return aligned, rot

def zoomout_after_align(image, pad_ratio=0.0, keep_canvas=False):
    """Pad viền đen sau khi align. 
    - keep_canvas=False: pad rồi resize về kích cỡ cũ (giống hành vi cũ)
    - keep_canvas=True : trả về khổ ảnh to hơn (phình canvas)
    """
    if pad_ratio <= 1e-6:
        return image, (0, 0)

    h, w = image.shape[:2]
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    padded = cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    if keep_canvas:
        return padded, (pad_x, pad_y)
    else:
        resized = cv2.resize(padded, (w, h))
        return resized, (pad_x, pad_y)

def load_board_model(model_path="runs/segment/yolov8_segment_board/weights/best.pt"):
    return YOLO(model_path)

# =============================
# YOLO detect
# =============================
def detect_board(model, frame, expand_ratio=1.0):
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
        aligned, rot = align_board(
            frame_resized, found_quad, output_size=(640, 640), expand_ratio=expand_ratio
        )
        aligned_zoomout, (pad_x, pad_y) = zoomout_after_align(
            aligned, pad_ratio=PAD_RATIO, keep_canvas=False
        )
        grid_info = {
            "pad_x": pad_x,
            "pad_y": pad_y,
            "usable_w": 640 - 2 * pad_x if pad_x else 640,
            "usable_h": 640 - 2 * pad_y if pad_y else 640,
            "rot": int(rot),  # 0 hoặc 90
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

def _detect_geom_quad_and_warp(frame, expand_ratio=1.0):
    global _last_warped, _last_quad, _last_mask, _stable_counter

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    blurred = cv2.medianBlur(filtered, 7)

    h, w = blurred.shape
    margin = 0.10
    y0, y1 = int(h * margin), int(h * (1 - margin))
    x0, x1 = int(w * margin), int(w * (1 - margin))
    roi = blurred[y0:y1, x0:x1]
    roi_offset = (x0, y0)

    edges = cv2.Canny(roi, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_rect = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4000:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2) + np.array(roi_offset, dtype=np.float32)
            if _is_rectangle(pts):
                best_rect = pts
                break

    if best_rect is not None:
        ordered = _order_points_rect(best_rect)
        ordered = expand_quad(ordered, expand_ratio)  # nới rộng giống YOLO

        (tl, tr, br, bl) = ordered
        width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
        height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
        dst = np.array([[0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(frame, M, (int(width), int(height)))

        _last_warped = warped
        _last_quad = ordered
        _last_mask = edges
        _stable_counter = _STABLE_FRAMES

        aligned = cv2.resize(warped, (640, 640))
        aligned, rot = _auto_rotate_long_side_vertical(aligned, ordered)
        aligned_zoomout, (pad_x, pad_y) = zoomout_after_align(
            aligned, pad_ratio=PAD_RATIO, keep_canvas=False
        )
        grid_info = {
            "pad_x": pad_x,
            "pad_y": pad_y,
            "usable_w": 640 - 2 * pad_x if pad_x else 640,
            "usable_h": 640 - 2 * pad_y if pad_y else 640,
            "rot": int(rot),
        }
        return aligned_zoomout, True, grid_info

    if _stable_counter > 0 and _last_warped is not None:
        _stable_counter -= 1
        aligned = cv2.resize(_last_warped, (640, 640))
        aligned, rot = _auto_rotate_long_side_vertical(aligned, _last_quad)
        aligned_zoomout, (pad_x, pad_y) = zoomout_after_align(
            aligned, pad_ratio=PAD_RATIO, keep_canvas=False
        )
        grid_info = {
            "pad_x": pad_x,
            "pad_y": pad_y,
            "usable_w": 640 - 2 * pad_x if pad_x else 640,
            "usable_h": 640 - 2 * pad_y if pad_y else 640,
            "rot": int(rot),
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
def detect_board_with_mode(frame, mode="yolo", model=None, model_light=None, model_heavy=None, expand_ratio=1.0):
    """
    mode:
      - 'yolo' : dùng YOLO duy nhất (ưu tiên model_light nếu có, nếu không dùng 'model')
      - 'geom' : pipeline hình học (nhẹ)
      - 'auto' : ưu tiên nhẹ trước: geom -> YOLO nhẹ -> YOLO nặng
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame phải là numpy.ndarray (H, W, 3) BGR")

    # Chuẩn hóa tham số tương thích cũ
    if model_light is None and model is not None:
        model_light = model

    if mode == "yolo":
        use_model = model_light if model_light is not None else model_heavy
        if use_model is None:
            raise ValueError("Cần truyền model YOLO (model_light hoặc model_heavy) khi dùng mode 'yolo'")
        return detect_board(use_model, frame, expand_ratio=expand_ratio)

    elif mode == "geom":
        return _detect_geom_quad_and_warp(frame, expand_ratio=expand_ratio)

    elif mode == "auto":
        # 1) Ưu tiên pipeline nhẹ
        img, found, info = _detect_geom_quad_and_warp(frame, expand_ratio=expand_ratio)
        if found:
            return img, found, info

        # 2) YOLO nhẹ (nếu có)
        if model_light is not None:
            img, found, info = detect_board(model_light, frame, expand_ratio=expand_ratio)
            if found:
                return img, found, info

        # 3) YOLO nặng (nếu có)
        if model_heavy is not None:
            img, found, info = detect_board(model_heavy, frame, expand_ratio=expand_ratio)
            if found:
                return img, found, info

        # 4) Fallback cuối
        frame_resized = cv2.resize(frame, (640, 640))
        return frame_resized, False, {"pad_x": 0, "pad_y": 0, "usable_w": 0, "usable_h": 0, "rot": 0}

    else:
        raise ValueError("Mode không hợp lệ. Chọn 'yolo', 'geom' hoặc 'auto'.")

# =============================
# Demo đơn giản (đọc 1 ảnh, in ra bàn cờ đã warp)
# =============================
if __name__ == "__main__":
    # Đọc ảnh
    image_path = "ban_co_mau.jpg"  # đổi thành ảnh của bạn
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    # Nạp YOLO (nếu dùng yolo/auto)
    model = load_board_model("runs/segment/yolov8_segment_board/weights/best.pt")

    # Gọi detect + warp (nới rộng khung ~6%)
    warped, found, info = detect_board_with_mode(
        img, mode="yolo", model=model, expand_ratio=EXPAND_RATIO
    )

    print("Found:", found)
    print("Grid info:", info)

    # Hiển thị
    cv2.imshow("Warped Board", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu ảnh
    cv2.imwrite("warped_output.jpg", warped)
