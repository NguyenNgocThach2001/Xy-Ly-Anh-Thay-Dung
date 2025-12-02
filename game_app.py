import os
from datetime import datetime

import cv2
import pygame
import sys
import numpy as np
import threading
import time

from detection.board_detection import load_board_model, detect_board_with_mode, warp_with_quad
from detection.piece_detection import load_piece_model, detect_pieces_and_get_positions
from pygame_board.board_display import draw_board, draw_pieces
# ĐÃ ĐỔI: import DebugPanel từ record_panel để có thể xóa debug_utils.py
from pygame_board.record_panel import DebugPanel
# LẤY THÊM util grid để dựng trạng thái replay trên board chính
from pygame_board.pygame_board import PygameBoard, grid_from_pieces, apply_move

# BỎ history_panel; Replay tự quản lý logic
from pygame_board.replay_panel import (
    handle_replay_click,
    draw_replay_tab,
    get_replay_state,   # <-- NEW: lấy moves/step/auto cho render
)
from pygame_board.record_panel import (
    handle_record_click,
    draw_record_tab,
)

# =============================
# Constants (UI mượt, camera thread riêng 10 FPS)
# =============================
DEFAULT_FPS = 15  # giảm từ 120 -> 60 cho mượt + nhẹ
PIECE_CONFIDENCE = 0.6
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
FRAME_GRAB_EVERY = 1
CAMERA_CANDIDATES = [0, 1, 2, 3]
FPS_OPTIONS = [5, 10, 15, 25, 30, 60, 90, 120]
FRAME_GRAB_OPTIONS = [1, 2, 3, 5, 10]
PIECE_CONF_OPTIONS = [0.5, 0.6, 0.7, 0.8]

LEFT_COL_RATIO = 0.58  # tỉ lệ cột trái (Board) trong vùng view
TOPBAR_H = 40          # thanh top đầu màn hình có 2 tab Record/Replay

# Custom font for pieces / UI
CUSTOM_FONT_FILE = "NotoSansSC-VariableFont_wght.ttf"


def load_app_font(size: int) -> pygame.font.Font:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(base_dir, CUSTOM_FONT_FILE)
        if os.path.exists(font_path):
            return pygame.font.Font(font_path, size)
    except Exception:
        pass

    fallback_names = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    for name in fallback_names:
        try:
            return pygame.font.SysFont(name, size)
        except Exception:
            continue

    return pygame.font.Font(None, size)

# =============================
# Camera reader thread + Warp (camera = 10 FPS)
# =============================
class CameraReader:
    """
    Đọc camera trên thread riêng và xử lý:
      - Xoay frame theo rotate_steps
      - Warp với board_quad khi board_locked
    Kết quả giữ sẵn:
      - latest_rotated (BGR)
      - latest_warp (BGR) + latest_grid_info
      - latest_seq (tăng mỗi khi có frame xử lý xong)
    Lấy ra bằng get_*() — non-blocking.
    """
    def __init__(self, index: int, width: int, height: int, target_fps: int = 10):
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.target_fps = max(1, int(target_fps))  # ép 10 FPS

        self._cap = None
        self._lock = threading.Lock()

        self._latest_rotated = None
        self._latest_warp = None
        self._latest_grid_info = None
        self._latest_seq = 0  # sequence id cho frame/warp mới

        self._running = False
        thead = None
        self._thread = None

        # runtime state
        self.ready = False
        self._last_ok_ts = 0.0
        self._reopen_requested = False

        # params cập nhật từ main thread
        self._rotate_steps = 0
        self._board_quad = None
        self._board_locked = False

    # ---- control setters ----
    def set_index(self, index: int):
        index = int(index)
        if index != self.index:
            self.index = index
            self._reopen_requested = True

    def set_rotation(self, steps: int):
        self._rotate_steps = int(steps) % 4

    def set_quad(self, quad):
        # quad: np.ndarray shape (4,2) float32 or None
        self._board_quad = quad

    def set_board_locked(self, locked: bool):
        self._board_locked = bool(locked)

    # ---- lifecycle ----
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=0.5)
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    # ---- getters (non-blocking, copy) ----
    def get_frame(self):
        with self._lock:
            return None if self._latest_rotated is None else self._latest_rotated.copy()

    def get_warp_and_info(self):
        with self._lock:
            if self._latest_warp is None:
                return None, None
            return self._latest_warp.copy(), dict(self._latest_grid_info or {})

    # Kèm sequence id để biết có frame mới hay không
    def get_warp_and_info_seq(self):
        with self._lock:
            if self._latest_warp is None:
                return None, None, self._latest_seq
            return self._latest_warp.copy(), dict(self._latest_grid_info or {}), self._latest_seq

    # ---- internals ----
    def _open_cap(self):
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = cv2.VideoCapture(self.index)
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            pass

    def _loop(self):
        self._open_cap()
        min_sleep = 1.0 / float(self.target_fps)  # 10 FPS
        while self._running:
            if self._reopen_requested:
                self._reopen_requested = False
                self._open_cap()

            ok, frame = (False, None)
            try:
                if self._cap is not None:
                    ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None

            now = time.time()
            if ok and frame is not None:
                # rotate
                rs = self._rotate_steps
                if rs == 1:
                    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rs == 2:
                    rotated = cv2.rotate(frame, cv2.ROTATE_180)
                elif rs == 3:
                    rotated = cv2.ROTATE_90_CLOCKWISE
                    rotated = cv2.rotate(frame, rotated)
                else:
                    rotated = frame

                # warp nếu đã khoá bàn và có quad
                warped = None
                grid_info = None
                if self._board_locked and self._board_quad is not None:
                    try:
                        warped, grid_info = warp_with_quad(rotated, self._board_quad)
                    except Exception:
                        warped, grid_info = None, None

                with self._lock:
                    self._latest_rotated = rotated
                    self._latest_warp = warped
                    self._latest_grid_info = grid_info
                    self._latest_seq += 1  # đánh dấu có frame/warp mới

                self._last_ok_ts = now
                self.ready = True
                time.sleep(min_sleep)  # throttle cứng 10 FPS
            else:
                if now - self._last_ok_ts > 1.0:
                    self.ready = False
                time.sleep(0.05)  # ngủ lâu hơn khi lỗi để bớt CPU

# =============================
# Helpers
# =============================
def _as_int(value, default=0):
    try:
        if isinstance(value, (list, tuple)):
            if not value:
                return int(default)
            value = value[0]
        return int(value)
    except Exception:
        return int(default)

def flip_pieces_180(pieces):
    return [(name, (8 - c, 9 - r)) for name, (c, r) in pieces]

def detect_and_orient_pieces(board_img, grid_info, piece_model, piece_conf, flip_view):
    """
    Trả về (pieces, piece_image). Nếu flip_view=True -> lật 180° cho khớp hướng UI.
    """
    if board_img is None or grid_info is None:
        return [], None

    res = detect_pieces_and_get_positions(piece_model, board_img, grid_info, conf=piece_conf)

    pieces, piece_image = [], None
    if isinstance(res, tuple):
        if len(res) >= 1:
            pieces = res[0] if res[0] is not None else []
        if len(res) >= 2:
            piece_image = res[1]
    elif isinstance(res, list):
        pieces = res
    elif res is None:
        pieces = []
    else:
        try:
            pieces = list(res)
        except Exception:
            pieces = []

    if flip_view and pieces:
        pieces = flip_pieces_180(pieces)
    return pieces, piece_image

def _order_quad(pts4):
    pts = np.array(pts4, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _get_quad_from_info(info):
    if not isinstance(info, dict):
        return None
    for key in ("quad", "corners", "points", "board_quad"):
        if key in info and key is not None and info[key] is not None:
            try:
                q = _order_quad(info[key])
                if q.shape == (4, 2):
                    return q
            except Exception:
                pass
    return None

# =============================
# Vẽ overlay lưới MANUAL (hiển thị)
# - Giữ kích thước ô cố định; tăng tổng kích thước theo river_extra.
# - Hiển thị 2 đường biên khe sông (hai vạch đỏ).
# =============================
def _draw_manual_grid_overlay(surf, gi, man, size_tuple):
    """
    surf: pygame.Surface (đã có alpha)
    gi:   grid_info từ warp_with_quad (pad_x/pad_y/usable_w/usable_h/rot...)
    man:  {swap_axes, river_extra, scale_x, scale_y, offset_x, offset_y, visible}
    size_tuple: (w, h) của surface đích (preview đã scale)
    """
    if gi is None or not man.get("visible", True):
        return

    w, h = size_tuple
    sx = w / 640.0
    sy = h / 640.0

    pad_x = float(gi.get("pad_x", 0)) * sx
    pad_y = float(gi.get("pad_y", 0)) * sy
    usable_w = float(gi.get("usable_w", w)) * sx
    usable_h = float(gi.get("usable_h", h)) * sy

    swap_axes   = bool(man.get("swap_axes", False))
    river_extra = max(0.0, float(man.get("river_extra", 0.0)))
    scale_x     = float(man.get("scale_x", 1.0))
    scale_y     = float(man.get("scale_y", 1.0))
    off_x       = float(man.get("offset_x", 0.0)) * sx
    off_y       = float(man.get("offset_y", 0.0)) * sy

    # Kích thước nền (chưa cộng sông)
    base_w = usable_w * max(0.2, min(5.0, scale_x))
    base_h = usable_h * max(0.2, min(5.0, scale_y))

    x0 = pad_x + off_x
    y0 = pad_y + off_y

    GRID = (40, 40, 40, 170)
    RIVER = (220, 50, 50, 200)
    OUTLINE = (10, 10, 10, 200)

    if not swap_axes:
        # Ô giữ nguyên; biên dưới phải nằm ở hàng 9
        cell_x = base_w / 8.0     # 9 cột => 8 khoảng
        cell_y = base_h / 9.0     # 10 hàng => 9 khoảng (không tính sông)
        y_river_top = y0 + 4.0 * cell_y
        y_river_bot = y_river_top + river_extra

        # Biên dưới thực tế (đường hàng 9)
        y_bottom = y0 + 8.0 * cell_y + river_extra
        # Khung ngoài: rộng = base_w, cao = (8*cell_y + river_extra)
        pygame.draw.rect(surf, OUTLINE,
                         pygame.Rect(int(x0), int(y0), int(base_w), int(y_bottom - y0)), 2)

        # Cột (đi hết từ top tới đúng hàng 9)
        for i in range(9):
            x = int(round(x0 + i * cell_x))
            pygame.draw.line(surf, GRID, (x, int(y0)), (x, int(y_bottom)), 1)

        # Hàng phía trên sông (0..4)
        for r in range(5):
            y = int(round(y0 + r * cell_y))
            pygame.draw.line(surf, GRID, (int(x0), y), (int(x0 + base_w), y), 1)

        # Hàng phía dưới sông (6..9) — đẩy xuống thêm river_extra
        for r in range(6, 10):
            y = int(round(y_river_bot + (r - 5) * cell_y))
            pygame.draw.line(surf, GRID, (int(x0), y), (int(x0 + base_w), y), 1)

        # Hai đường sông (đỏ)
        pygame.draw.line(surf, RIVER, (int(x0), int(round(y_river_top))),
                         (int(x0 + base_w), int(round(y_river_top))), 2)
        pygame.draw.line(surf, RIVER, (int(x0), int(round(y_river_bot))),
                         (int(x0 + base_w), int(round(y_river_bot))), 2)

    else:
        # Hoán trục: biên phải nằm ở cột 9
        cell_x = base_w / 9.0     # 10 hàng theo trục X (không tính sông)
        cell_y = base_h / 8.0     # 9 cột theo trục Y
        x_river_left = x0 + 4.0 * cell_x
        x_river_right = x_river_left + river_extra

        # Biên phải thực tế (đường cột 9)
        x_right = x0 + 8.0 * cell_x + river_extra
        # Khung ngoài: cao = base_h, rộng = (8*cell_x + river_extra)
        pygame.draw.rect(surf, OUTLINE,
                         pygame.Rect(int(x0), int(y0), int(x_right - x0), int(base_h)), 2)

        # Hàng ngang (đi hết từ trái tới biên phải thực tế)
        for i in range(9):
            y = int(round(y0 + i * cell_y))
            pygame.draw.line(surf, GRID, (int(x0), y), (int(x_right), y), 1)

        # Cột dọc phía trước sông (0..4)
        for c in range(5):
            x = int(round(x0 + c * cell_x))
            pygame.draw.line(surf, GRID, (x, int(y0)), (x, int(y0 + base_h)), 1)

        # Cột dọc phía sau sông (6..9) — đẩy sang phải thêm river_extra
        for c in range(6, 10):
            x = int(round(x_river_right + (c - 5) * cell_x))
            pygame.draw.line(surf, GRID, (x, int(y0)), (x, int(y0 + base_h)), 1)

        # Hai đường sông (đỏ)
        pygame.draw.line(surf, RIVER, (int(round(x_river_left)), int(y0)),
                         (int(round(x_river_left)), int(y0 + base_h)), 2)
        pygame.draw.line(surf, RIVER, (int(round(x_river_right)), int(y0)),
                         (int(round(x_river_right)), int(y0 + base_h)), 2)
# =============================
# Event handling
# =============================
def handle_events(
    event,
    running,
    active_tab,
    ui,
    debug_panel,
    num_available_cameras,
    available_cameras,
    replay_moves,
    replay_index,
    replay_auto,
    record_filename,
    camera_ready,
    camera_status,
    recording,
    record_status,
    fps_dropdown_open,
    fps_current,
    frame_grab_dropdown_open,
    frame_grab_every_current,
    camera_dropdown_open,
    piece_conf_dropdown_open,
    piece_conf_current,
    debug_enabled,
    selected_camera_index,
    rotate_steps,
    # Segment
    segment_requested,
    rect_board_area,
    rect_right_panel,
    rect_tabs,
    rect_dropdowns,
    rect_buttons,
    rect_lists,
    rect_playback,
    rect_top_tabs,   # <-- NEW: top bar rects
    # Manual grid
    manual_enabled,
    manual_swap_axes,
    manual_river_extra,
    manual_scale_x,
    manual_scale_y,
    manual_offset_x,
    manual_offset_y,
    # NEW: flip UI view
    ui_flip_view,
):
    if event.type == pygame.QUIT:
        running = False
        return (
            running, active_tab, ui, debug_panel, num_available_cameras, available_cameras,
            replay_moves, replay_index, replay_auto, record_filename, camera_ready, camera_status, recording,
            record_status, fps_dropdown_open, fps_current, frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current, debug_enabled, selected_camera_index,
            rotate_steps, segment_requested, manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y, ui_flip_view
        )

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            running = False
        elif event.key == pygame.K_TAB:
            active_tab = (active_tab + 1) % 2  # CHỈ 2 TAB: Record / Replay
        elif event.key == pygame.K_r:
            rotate_steps = (rotate_steps + 1) % 4

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        mx, my = event.pos

        # TOP BAR tabs
        if rect_top_tabs["record"].collidepoint(mx, my):
            active_tab = 0
        elif rect_top_tabs["replay"].collidepoint(mx, my):
            active_tab = 1

        # PANEL tabs (giữ như cũ)
        if rect_tabs["record"].collidepoint(mx, my):
            active_tab = 0
        elif rect_tabs["replay"].collidepoint(mx, my):
            active_tab = 1

        if active_tab == 0:
            prev_recording = recording
            clicked_record_button = rect_buttons["record"].collidepoint(mx, my)

            result = handle_record_click(
                mx=mx,
                my=my,
                test_piece_button_rect=rect_buttons["testpiece"],
                rotate_button_rect=rect_buttons["rotate"],
                camera_button_rect=rect_buttons["checkcam"],
                segment_button_rect=rect_buttons["segment"],
                record_button_rect=rect_buttons["record"],
                fps_dropdown_rect=rect_dropdowns["fps"],
                grab_dropdown_rect=rect_dropdowns["grab"],
                camera_dropdown_rect=rect_dropdowns["camera"],
                piece_conf_dropdown_rect=rect_dropdowns["pconf"],
                debug_button_rect=rect_buttons["debug"],
                camera_indices=available_cameras,
                cap=None,
                camera_ready=camera_ready,
                camera_status=camera_status,
                recording=recording,
                record_status=record_status,
                fps_dropdown_open=fps_dropdown_open,
                fps_current=fps_current,
                frame_grab_dropdown_open=frame_grab_dropdown_open,
                frame_grab_every_current=frame_grab_every_current,
                camera_dropdown_open=camera_dropdown_open,
                piece_conf_dropdown_open=piece_conf_dropdown_open,
                piece_conf_current=piece_conf_current,
                debug_enabled=debug_enabled,
                selected_camera_index=selected_camera_index,
                fps_options=FPS_OPTIONS,
                frame_grab_options=FRAME_GRAB_OPTIONS,
                piece_conf_options=PIECE_CONF_OPTIONS,
                rotate_steps=rotate_steps,
                # manual grid controls & state
                manual_toggle_rect=rect_buttons.get("manual_toggle"),
                swap_axes_rect=rect_buttons.get("swap_axes"),
                river_minus_rect=rect_buttons.get("river_minus"),
                river_plus_rect=rect_buttons.get("river_plus"),
                scx_minus_rect=rect_buttons.get("scx_minus"),
                scx_plus_rect=rect_buttons.get("scx_plus"),
                scy_minus_rect=rect_buttons.get("scy_minus"),
                scy_plus_rect=rect_buttons.get("scy_plus"),
                offx_minus_rect=rect_buttons.get("offx_minus"),
                offx_plus_rect=rect_buttons.get("offx_plus"),
                offy_minus_rect=rect_buttons.get("offy_minus"),
                offy_plus_rect=rect_buttons.get("offy_plus"),
                manual_enabled=manual_enabled,
                manual_swap_axes=manual_swap_axes,
                manual_river_extra=manual_river_extra,
                manual_scale_x=manual_scale_x,
                manual_scale_y=manual_scale_y,
                manual_offset_x=manual_offset_x,
                manual_offset_y=manual_offset_y,
            )
            (
                cap_dummy, camera_ready, camera_status, recording, record_status,
                fps_dropdown_open, fps_current,
                frame_grab_dropdown_open, frame_grab_every_current,
                camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
                debug_enabled, selected_camera_index, rotate_steps,
                segment_clicked,
                manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
                manual_offset_x, manual_offset_y
            ) = result

            if clicked_record_button and prev_recording and not recording:
                try:
                    ui.reset()
                except Exception:
                    pass

            rotate_steps = _as_int(rotate_steps, 0)

            if segment_clicked:
                segment_requested = True
                recording = False
                record_status = "Idle"

            if rect_buttons.get("uiflip") and rect_buttons["uiflip"].collidepoint(mx, my):
                ui_flip_view = not ui_flip_view

        elif active_tab == 1:
            replay_moves, replay_index, replay_auto = handle_replay_click(
                mx, my, rect_playback, replay_moves, replay_index, replay_auto
            )

    if event.type == pygame.USEREVENT and getattr(event, "name", None) == "TEST_PIECE":
        segment_requested = ("__TEST_PIECE__",)
    return (
        running, active_tab, ui, debug_panel, num_available_cameras, available_cameras,
        replay_moves, replay_index, replay_auto, record_filename, camera_ready, camera_status, recording,
        record_status, fps_dropdown_open, fps_current, frame_grab_dropdown_open, frame_grab_every_current,
        camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current, debug_enabled, selected_camera_index,
        rotate_steps, segment_requested, manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
        manual_offset_x, manual_offset_y, ui_flip_view
    )

# =============================
# Right panel drawing
# =============================
def draw_right_panel(
    screen,
    font,
    active_tab,
    ui,
    replay_moves,
    replay_index,
    replay_auto,
    record_filename,
    camera_ready,
    camera_status,
    recording,
    record_status,
    fps_dropdown_open,
    fps_current,
    frame_grab_dropdown_open,
    frame_grab_every_current,
    camera_dropdown_open,
    piece_conf_dropdown_open,
    piece_conf_current,
    debug_enabled,
    selected_camera_index,
    rect_right_panel,
    rect_tabs,
    rect_dropdowns,
    rect_buttons,
    rect_lists,
    rect_playback,
    # preview context
    processing_frame,
    last_board_img,
    pieces_preview,
    has_board_preview,
    clock,
    board_w,
    board_locked,
    # panel
    debug_panel,
    available_cameras,
    # manual grid state
    manual_enabled,
    manual_swap_axes,
    manual_river_extra,
    manual_scale_x,
    manual_scale_y,
    manual_offset_x,
    manual_offset_y,
    # NEW
    ui_flip_view,
):
    pygame.draw.rect(screen, (30, 30, 30), rect_tabs["bar"])
    for key, label in [("record", "Record"), ("replay", "Replay")]:
        pygame.draw.rect(screen, (60, 60, 60), rect_tabs[key])
        text = font.render(label, True, (255, 255, 255))
        screen.blit(text, (rect_tabs[key].x + 10, rect_tabs[key].y + 8))

    pygame.draw.rect(screen, (40, 40, 40), rect_right_panel)

    if active_tab == 0:
        rs = getattr(ui, "rotate_steps", 0)
        rs = _as_int(rs, 0)

        draw_record_tab(
            screen=screen,
            test_piece_button_rect=rect_buttons["testpiece"],
            panel=debug_panel,
            frame=processing_frame,
            last_board_img=last_board_img,
            pieces=pieces_preview,
            has_board=has_board_preview,
            clock=clock,
            board_w=board_w,
            panel_origin_y=rect_tabs["bar"].bottom,
            mono=font,
            rotate_button_rect=rect_buttons["rotate"],
            camera_button_rect=rect_buttons["checkcam"],
            segment_button_rect=rect_buttons["segment"],
            record_button_rect=rect_buttons["record"],
            fps_dropdown_rect=rect_dropdowns["fps"],
            grab_dropdown_rect=rect_dropdowns["grab"],
            camera_dropdown_rect=rect_dropdowns["camera"],
            piece_conf_dropdown_rect=rect_dropdowns["pconf"],
            debug_button_rect=rect_buttons["debug"],
            camera_indices=available_cameras,
            camera_ready=camera_ready,
            debug_enabled=debug_enabled,
            recording=recording,
            fps_dropdown_open=fps_dropdown_open,
            fps_current=fps_current,
            frame_grab_dropdown_open=frame_grab_dropdown_open,
            frame_grab_every_current=frame_grab_every_current,
            camera_dropdown_open=camera_dropdown_open,
            selected_camera_index=selected_camera_index,
            piece_conf_dropdown_open=piece_conf_dropdown_open,
            piece_conf_current=piece_conf_current,
            camera_status=camera_status,
            record_status=record_status,
            fps_options=FPS_OPTIONS,
            frame_grab_options=FRAME_GRAB_OPTIONS,
            piece_conf_options=PIECE_CONF_OPTIONS,
            rotate_steps=rs,
            board_locked=board_locked,
            # manual controls + state
            manual_toggle_rect=rect_buttons.get("manual_toggle"),
            swap_axes_rect=rect_buttons.get("swap_axes"),
            river_minus_rect=rect_buttons.get("river_minus"),
            river_plus_rect=rect_buttons.get("river_plus"),
            scx_minus_rect=rect_buttons.get("scx_minus"),
            scx_plus_rect=rect_buttons.get("scx_plus"),
            scy_minus_rect=rect_buttons.get("scy_minus"),
            scy_plus_rect=rect_buttons.get("scy_plus"),
            offx_minus_rect=rect_buttons.get("offx_minus"),
            offx_plus_rect=rect_buttons.get("offx_plus"),
            offy_minus_rect=rect_buttons.get("offy_minus"),
            offy_plus_rect=rect_buttons.get("offy_plus"),
            manual_enabled=manual_enabled,
            manual_swap_axes=manual_swap_axes,
            manual_river_extra=manual_river_extra,
            manual_scale_x=manual_scale_x,
            manual_scale_y=manual_scale_y,
            manual_offset_x=manual_offset_x,
            manual_offset_y=manual_offset_y,
            # NEW: truyền filename + khung panel để vẽ Move Log
            record_filename=record_filename,
            right_panel_rect=rect_right_panel,
        )

        # NEW small button: flip UI view
        btn = rect_buttons.get("uiflip")
        if btn:
            color = (60, 60, 60)
            mp = pygame.mouse.get_pos()
            if btn.collidepoint(mp):
                color = (80, 80, 80)
            pygame.draw.rect(screen, color, btn, border_radius=6)
            pygame.draw.rect(screen, (120, 120, 120), btn, 1, border_radius=6)
            label = "UI: Red bottom" if ui_flip_view else "UI: Red top"
            text = font.render(label, True, (255, 255, 255))
            screen.blit(text, (btn.x + 8, btn.y + 6))

    else:
        draw_replay_tab(screen, font, replay_moves, replay_index, replay_auto, record_filename, rect_playback)

# =============================
# Main app
# =============================
def run():
    pygame.init()
    pygame.font.init()

    screen_info = pygame.display.Info()
    W, H = screen_info.current_w, screen_info.current_h
    # thử bật vsync nếu môi trường hỗ trợ SDL2
    try:
        screen = pygame.display.set_mode((W, H), vsync=1)
    except Exception:
        screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Chinese Chess App")

    font = load_app_font(20)
    font_small = load_app_font(16)  # cache font nhỏ, tránh tạo lại mỗi frame
    debug_panel = DebugPanel(screen)

    # ======= Top bar (2 tab Record / Replay) =======
    rect_top_tabs = {
        "bar": pygame.Rect(0, 0, W, TOPBAR_H),
        "record": pygame.Rect(10, 6, 120, TOPBAR_H - 12),
        "replay": pygame.Rect(140, 6, 120, TOPBAR_H - 12),
    }

    # ======= Layout chính nằm dưới top-bar =======
    content_h = H - TOPBAR_H
    left_w = int(W * 0.68)
    right_w = W - left_w
    top_h = int(content_h * 0.08)  # chiều cao thanh tab trong panel phải
    view_area = pygame.Rect(0, TOPBAR_H, left_w, content_h)
    right_panel = pygame.Rect(left_w, TOPBAR_H, right_w, content_h)

    # Tabs (CHỈ 2 TAB) trong panel phải như cũ
    tab_bar = pygame.Rect(right_panel.x, right_panel.y, right_panel.width, top_h)
    tab_w = right_w // 2
    rect_tabs = {
        "bar": tab_bar,
        "record": pygame.Rect(right_panel.x, right_panel.y, tab_w, top_h),
        "replay": pygame.Rect(right_panel.x + 1 * tab_w, right_panel.y, tab_w, top_h),
    }

    # Dropdowns & buttons
    dd_w = right_w - 40
    x0 = right_panel.x + 20
    y0 = right_panel.y + top_h + 20
    dd_h = 30
    rect_dropdowns = {
        "fps": pygame.Rect(x0, y0, dd_w, dd_h),
        "grab": pygame.Rect(x0, y0 + 50, dd_w, dd_h),
        "camera": pygame.Rect(x0, y0 + 100, dd_w, dd_h),
        "pconf": pygame.Rect(x0, y0 + 150, dd_w, dd_h),
    }
    btn_w, btn_h = (dd_w - 10) // 2, 36
    btn_y = y0 + 210
    rect_buttons = {
        "rotate": pygame.Rect(x0, btn_y, btn_w, btn_h),
        "checkcam": pygame.Rect(x0 + btn_w + 10, btn_y, btn_w, btn_h),
        "segment": pygame.Rect(x0, btn_y + 50, dd_w, btn_h),
        "record": pygame.Rect(x0, btn_y + 100, dd_w, btn_h),
        "debug": None,
        "testpiece": pygame.Rect(x0, btn_y + 200, dd_w, btn_h),
    }

    # NEW: UI flip button (đặt ngay dưới Test piece)
    rect_buttons["uiflip"] = pygame.Rect(x0, btn_y + 250, dd_w, btn_h)

    # Manual Grid control rows
    mg_y = btn_y + 300  # đẩy xuống 1 hàng vì thêm nút uiflip
    mg_gap = 8
    rect_buttons["manual_toggle"] = pygame.Rect(x0, mg_y, dd_w, btn_h)
    rect_buttons["swap_axes"] = pygame.Rect(x0, mg_y + (btn_h + mg_gap), dd_w, btn_h)

    # River
    rect_buttons["river_minus"] = pygame.Rect(x0, mg_y + 2*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    rect_buttons["river_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 2*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    # Scale X
    rect_buttons["scx_minus"] = pygame.Rect(x0, mg_y + 3*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    rect_buttons["scx_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 3*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    # Scale Y
    rect_buttons["scy_minus"] = pygame.Rect(x0, mg_y + 4*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    rect_buttons["scy_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 4*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    # NEW Offset X
    rect_buttons["offx_minus"] = pygame.Rect(x0, mg_y + 5*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    rect_buttons["offx_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 5*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    # NEW Offset Y
    rect_buttons["offy_minus"] = pygame.Rect(x0, mg_y + 6*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    rect_buttons["offy_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 6*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)

    rect_lists = {"history": pygame.Rect(x0, y0, dd_w, content_h - (y0 - right_panel.y) - 40)}  # giữ placeholder
    rect_playback = {
        "prev": pygame.Rect(x0, y0, (dd_w - 10) // 2, dd_h),
        "next": pygame.Rect(x0 + (dd_w + 10) // 2, y0, (dd_w - 10) // 2, dd_h),
        "auto": pygame.Rect(x0, y0 + 50, dd_w, dd_h),
    }

    rect_right_panel = right_panel
    rect_board_area = view_area

    # States
    running = True
    clock = pygame.time.Clock()

    active_tab = 0
    rotate_steps = 0

    # Models
    board_model = load_board_model()
    piece_model = load_piece_model()

    # Detect available cameras
    camera_ready = False
    camera_status = "Checking cameras..."
    available_cameras = []
    for cam_idx in CAMERA_CANDIDATES:
        cap_test = cv2.VideoCapture(cam_idx)
        cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        ok, _ = cap_test.read()
        if ok:
            available_cameras.append(cam_idx)
        cap_test.release()

    selected_camera_index = available_cameras[0] if available_cameras else 0
    if available_cameras:
        camera_ready = True
        camera_status = f"Found {len(available_cameras)} camera(s). Using {selected_camera_index}"
    else:
        camera_ready = False
        camera_status = "No camera found"

    # Camera thread: target_fps=10 (theo yêu cầu)
    camera_thread = None
    if camera_ready:
        camera_thread = CameraReader(selected_camera_index, CAPTURE_WIDTH, CAPTURE_HEIGHT, target_fps=10)
        camera_thread.set_rotation(rotate_steps)
        camera_thread.start()

    # Board segmentation
    board_locked = False
    board_quad = None
    last_board_img = None     # warped preview (từ thread)
    last_grid_info = None

    # Piece detection
    piece_conf_current = PIECE_CONFIDENCE

    # UI (KHỞI TẠO TRỐNG + BẬT VALIDATOR)
    ui = PygameBoard(draw_board, draw_pieces, font, init_full=False, use_validator=True)
    ui.rotate_steps = rotate_steps
    ui.debug_panel = debug_panel
    debug_enabled = True  # LUÔN BẬT Debug

    # Replay / record
    replay_moves = []
    replay_index = 0
    replay_auto = False
    recording = False
    record_filename = ""
    record_status = "Idle"

    # Dropdown flags/values
    fps_dropdown_open = False
    frame_grab_dropdown_open = False
    camera_dropdown_open = False
    piece_conf_dropdown_open = False

    fps_current = DEFAULT_FPS
    frame_grab_every_current = FRAME_GRAB_EVERY

    # Prepare plays dir
    os.makedirs("plays", exist_ok=True)

    frame_idx = 0
    last_grabbed = -999
    segment_requested = False

    # Cache preview surfaces
    input_preview_surface = None
    warped_preview_surface = None

    # One-shot Test Piece state
    test_piece_once_requested = False
    test_piece_overlay = None

    # Manual grid state
    # CHÚ Ý: manual_enabled bây giờ = CHỈ HIỂN THỊ overlay (show/hide).
    # Tham số manual LUÔN được áp dụng vào nhận dạng, dù overlay ẩn hay hiện.
    manual_enabled = False
    manual_swap_axes = False   # False: X->9 cột; True: Y->9 cột
    manual_river_extra = 64.0  # mặc định để luôn thấy 2 vạch
    manual_scale_x = 1.0
    manual_scale_y = 1.0
    manual_offset_x = 0.0      # px dịch lưới theo X
    manual_offset_y = 0.0      # px dịch lưới theo Y

    # Flip UI view (False = đỏ ở trên; True = đỏ ở dưới)
    ui_flip_view = False

    # utility: chuyển BGR np.ndarray -> pygame.Surface nhanh (ít copy)
    def bgr_to_surface_scaled(bgr_img, dest_w, dest_h):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        surf = pygame.image.frombuffer(rgb.data, (w, h), "RGB")
        if w != dest_w or h != dest_h:
            surf = pygame.transform.smoothscale(surf, (dest_w, dest_h))
        return surf

    # nhớ seq để detect đúng theo rate camera (khi Record)
    last_detect_seq = -1

    # ======= CACHE overlay lưới để không vẽ lại mỗi frame =======
    grid_overlay_cache = {
        "key": None,          # tuple các tham số
        "size": None,         # (w,h)
        "surface": None,      # pygame.Surface convert_alpha
    }

    def build_grid_key(gi: dict, man: dict, size_tuple):
        # Nếu overlay KHÔNG hiển thị -> off
        if gi is None or not man.get("visible", False):
            return ("off",)
        # chỉ lấy các field quan trọng (thiếu thì 0)
        k = (
            round(float(gi.get("pad_x", 0.0)), 2),
            round(float(gi.get("pad_y", 0.0)), 2),
            round(float(gi.get("usable_w", 640.0)), 2),
            round(float(gi.get("usable_h", 640.0)), 2),
            bool(man.get("swap_axes", False)),
            round(float(man.get("river_extra", 0.0)), 2),
            round(float(man.get("scale_x", 1.0)), 3),
            round(float(man.get("scale_y", 1.0)), 3),
            round(float(man.get("offset_x", 0.0)), 2),
            round(float(man.get("offset_y", 0.0)), 2),
            int(size_tuple[0]), int(size_tuple[1]),
        )
        return k

    def get_grid_overlay(gi, man, size_tuple):
        key = build_grid_key(gi, man, size_tuple)
        if key == ("off",):
            return None
        if (grid_overlay_cache["key"] != key) or (grid_overlay_cache["size"] != size_tuple) or (grid_overlay_cache["surface"] is None):
            # rebuild
            w, h = size_tuple
            surf = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()
            _draw_manual_grid_overlay(
                surf,
                gi,
                {
                    "visible": bool(man.get("visible", True)),
                    "swap_axes": bool(man.get("swap_axes", False)),
                    "river_extra": float(man.get("river_extra", 0.0)),
                    "scale_x": float(man.get("scale_x", 1.0)),
                    "scale_y": float(man.get("scale_y", 1.0)),
                    "offset_x": float(man.get("offset_x", 0.0)),
                    "offset_y": float(man.get("offset_y", 0.0)),
                },
                size_tuple,
            )
            grid_overlay_cache["key"] = key
            grid_overlay_cache["size"] = size_tuple
            grid_overlay_cache["surface"] = surf
        return grid_overlay_cache["surface"]

    def pieces_from_grid(grid):
        """Chuyển grid[r][c]=label -> list[(label,(c,r))] để vẽ trên board chính."""
        out = []
        for r, row in enumerate(grid):
            for c, name in enumerate(row):
                if name is not None:
                    out.append((name, (c, r)))
        return out

    # ======= helper: vẽ top bar tabs =======
    def draw_top_tabs(screen, font, rects, active):
        pygame.draw.rect(screen, (25, 25, 25), rects["bar"])
        for key, label in [("record", "Record"), ("replay", "Replay")]:
            r = rects[key]
            color = (60, 60, 60)
            if r.collidepoint(pygame.mouse.get_pos()):
                color = (80, 80, 80)
            if (active == 0 and key == "record") or (active == 1 and key == "replay"):
                color = (95, 95, 95)
            pygame.draw.rect(screen, color, r, border_radius=6)
            pygame.draw.rect(screen, (120, 120, 120), r, 1, border_radius=6)
            text = font.render(label, True, (255, 255, 255))
            screen.blit(text, (r.x + 10, r.y + 6))

    while running:
        # Events
        for event in pygame.event.get():
            (
                running, active_tab, ui, debug_panel, _num_cams, available_cameras,
                replay_moves, replay_index, replay_auto, record_filename, camera_ready, camera_status, recording,
                record_status, fps_dropdown_open, fps_current, frame_grab_dropdown_open, frame_grab_every_current,
                camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current, debug_enabled,
                selected_camera_index, rotate_steps, segment_requested,
                manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
                manual_offset_x, manual_offset_y, ui_flip_view
            ) = handle_events(
                event,
                running,
                active_tab,
                ui,
                debug_panel,
                len(available_cameras),
                available_cameras,
                replay_moves,
                replay_index,
                replay_auto,
                record_filename,
                camera_ready,
                camera_status,
                recording,
                record_status,
                fps_dropdown_open,
                fps_current,
                frame_grab_dropdown_open,
                frame_grab_every_current,
                camera_dropdown_open,
                piece_conf_dropdown_open,
                piece_conf_current,
                debug_enabled,
                selected_camera_index,
                rotate_steps,
                segment_requested,
                rect_board_area,
                rect_right_panel,
                rect_tabs,
                rect_dropdowns,
                rect_buttons,
                rect_lists,
                rect_playback,
                rect_top_tabs,  # NEW
                # manual params
                manual_enabled,
                manual_swap_axes,
                manual_river_extra,
                manual_scale_x,
                manual_scale_y,
                manual_offset_x,
                manual_offset_y,
                ui_flip_view,
            )

            if segment_requested == ("__TEST_PIECE__",):
                test_piece_once_requested = True
                segment_requested = False

        rotate_steps = _as_int(rotate_steps, 0)
        ui.rotate_steps = rotate_steps

        # Đồng bộ camera thread với UI (nhẹ)
        if camera_ready and camera_thread is not None:
            if camera_thread.index != selected_camera_index:
                camera_thread.set_index(selected_camera_index)
            camera_thread.set_rotation(rotate_steps)
            camera_thread.set_board_locked(board_locked)
            camera_thread.set_quad(board_quad)

        # Lấy frame xoay & warp từ thread (không tự warp ở main nữa)
        processing_frame = None
        if camera_thread is not None and (frame_idx - last_grabbed) >= frame_grab_every_current:
            pf = camera_thread.get_frame()
            if pf is not None:
                processing_frame = pf
                last_grabbed = frame_idx

        # SEGMENT (chỉ khi có frame mới)
        if segment_requested and processing_frame is not None:
            board_img, found, info = detect_board_with_mode(processing_frame, mode="auto", model=board_model)
            if found:
                q = _get_quad_from_info(info)
                if q is not None:
                    board_quad = q
                    # từ đây thread sẽ tự warp theo 10 FPS
                    last_board_img, last_grid_info = camera_thread.get_warp_and_info()
                else:
                    board_quad = None
                    last_board_img = None
                    last_grid_info = None

                board_locked = True
                camera_status = "Board segmented and locked."
            else:
                camera_status = "Segment failed. Try again."
                board_locked = False
                board_quad = None
                last_board_img = None
                last_grid_info = None
            segment_requested = False

        # PREVIEW: lấy warped từ thread (không warp lại)
        has_board_preview = False
        if board_locked and camera_thread is not None:
            wb, gi = camera_thread.get_warp_and_info()
            if wb is not None:
                last_board_img = wb
                last_grid_info = gi
            has_board_preview = last_board_img is not None

        # ONE-SHOT TEST PIECE (Preview only) — dùng warped từ thread
        if test_piece_once_requested and last_board_img is not None and last_grid_info is not None:
            try:
                gi = dict(last_grid_info or {})
                # LUÔN áp manual params (kể cả khi overlay ẩn)
                gi["swap_axes"] = bool(manual_swap_axes)
                gi["river_extra"] = float(manual_river_extra)
                gi["scale_x"] = float(manual_scale_x)
                gi["scale_y"] = float(manual_scale_y)
                gi["offset_x"] = float(manual_offset_x)
                gi["offset_y"] = float(manual_offset_y)

                test_pieces, _ = detect_and_orient_pieces(
                    last_board_img, gi, piece_model, piece_conf_current, ui_flip_view
                )
                test_piece_overlay = test_pieces or []
            except Exception:
                test_piece_overlay = []
            finally:
                test_piece_once_requested = False

        # RECORD PIPELINE — detect CHỈ khi có warp mới (đi đúng rate camera)
        if active_tab == 0 and recording and board_locked and camera_thread is not None:
            try:
                warped_img, grid_info, seq = camera_thread.get_warp_and_info_seq()
                if warped_img is not None and grid_info is not None:
                    if seq != last_detect_seq:  # chỉ detect khi có frame mới từ camera
                        last_detect_seq = seq

                        gi = dict(grid_info)
                        # LUÔN áp manual params (kể cả khi overlay ẩn)
                        gi["swap_axes"] = bool(manual_swap_axes)
                        gi["river_extra"] = float(manual_river_extra)
                        gi["scale_x"] = float(manual_scale_x)
                        gi["scale_y"] = float(manual_scale_y)
                        gi["offset_x"] = float(manual_offset_x)
                        gi["offset_y"] = float(manual_offset_y)

                        pieces, _ = detect_and_orient_pieces(
                            warped_img, gi, piece_model, piece_conf_current, ui_flip_view
                        )
                        ui.update_state(True, pieces)
                        if ui.has_new_move():
                            move_text = ui.get_last_move_text()
                            if move_text:
                                if not record_filename:
                                    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    record_filename = f"plays/play_{dt}.txt"
                                with open(record_filename, "a", encoding="utf-8") as f:
                                    f.write(move_text + "\n")
                # nếu không có warped mới → bỏ qua detect ở frame này
            except Exception:
                pass

        # Đang record thì bỏ overlay test
        if recording and test_piece_overlay:
            test_piece_overlay = None

        # ================== DRAW UI ==================
        screen.fill((20, 20, 20))

        # Top bar
        draw_top_tabs(screen, font, rect_top_tabs, active_tab)

        # View area boxes
        pygame.draw.rect(screen, (50, 50, 50), rect_board_area)

        margin = 24
        inner = rect_board_area.inflate(-2 * margin, -2 * margin)
        col_gap = 20

        left_w = int(inner.width * LEFT_COL_RATIO)
        right_w2 = inner.width - left_w - col_gap
        total_w = left_w + col_gap + right_w2
        container_x = inner.x + (inner.width - total_w) // 2

        left_col = pygame.Rect(container_x, inner.y, left_w, inner.height)
        right_col = pygame.Rect(left_col.right + col_gap, inner.y, right_w2, inner.height)

        board_aspect = 9 / 10
        fit_w = min(left_col.width, int(left_col.height * board_aspect))
        fit_h = int(fit_w / board_aspect)
        board_rect = pygame.Rect(
            left_col.x + (left_col.width - fit_w) // 2,
            left_col.y + (left_col.height - fit_h) // 2,
            fit_w,
            fit_h,
        )

        row_gap = 16
        row_h = (right_col.height - row_gap) // 2
        input_rect = pygame.Rect(right_col.x, right_col.y, right_col.width, row_h)
        warped_rect = pygame.Rect(right_col.x, input_rect.bottom + row_gap, right_col.width, right_col.height - row_h - row_gap)

        def _box(r):
            pygame.draw.rect(screen, (235, 235, 210), r, border_radius=6)
            pygame.draw.rect(screen, (90, 90, 90), r, 1, border_radius=6)

        _box(board_rect)
        _box(input_rect)
        _box(warped_rect)

        # Board UI (convert surface để blit nhanh)
        board_surface = pygame.Surface((board_rect.width, board_rect.height)).convert()
        board_surface.fill((235, 235, 210))
        draw_board(board_surface)

        # ---- RENDER TRẠNG THÁI LÊN BOARD CHÍNH ----
        if active_tab == 1:
            # Lấy state từ replay panel (đã có auto tick)
            moves, step, _auto = get_replay_state()
            # Dựng từ thế ban đầu của UI (đúng label 'red-*/black-*')
            start_pieces = ui._starting_position()
            grid = grid_from_pieces(start_pieces)
            # Áp các nước đi theo (sr,sc)->(dr,dc)
            for i in range(min(step, len(moves))):
                (sr, sc), (dr, dc) = moves[i]
                grid = apply_move(grid, (sr, sc), (dr, dc))
            render_state = pieces_from_grid(grid)
        else:
            # Record / live view
            render_state = ui.state_live

        if ui_flip_view and render_state:
            render_state = flip_pieces_180(render_state)

        draw_pieces(board_surface, font, render_state)

        # Overlay Test piece (chỉ khi không record và không ở replay)
        if (not recording) and (active_tab != 1) and test_piece_overlay:
            overlay = test_piece_overlay
            if ui_flip_view:
                overlay = flip_pieces_180(overlay)
            draw_pieces(board_surface, font, overlay)

        screen.blit(board_surface, (board_rect.x, board_rect.y))

        # Input preview (frame xoay từ thread)
        if processing_frame is not None:
            input_preview_surface = bgr_to_surface_scaled(processing_frame, input_rect.width, input_rect.height).convert()
        if input_preview_surface is not None:
            screen.blit(input_preview_surface, (input_rect.x, input_rect.y))

        # Warped preview (từ thread)
        if last_board_img is not None:
            warped_preview_surface = bgr_to_surface_scaled(last_board_img, warped_rect.width, warped_rect.height).convert()
        if warped_preview_surface is not None:
            screen.blit(warped_preview_surface, (warped_rect.x, warped_rect.y))
            if last_grid_info is not None:
                # Lấy overlay từ cache (convert_alpha để blit nhanh)
                man = {
                    "visible": manual_enabled,  # NÚT: chỉ hiện/ẩn
                    "swap_axes": manual_swap_axes,
                    "river_extra": manual_river_extra,
                    "scale_x": manual_scale_x,
                    "scale_y": manual_scale_y,
                    "offset_x": manual_offset_x,
                    "offset_y": manual_offset_y,
                }
                overlay_surface = get_grid_overlay(last_grid_info, man, (warped_rect.width, warped_rect.height))
                if overlay_surface is not None:
                    screen.blit(overlay_surface, (warped_rect.x, warped_rect.y))

        # Labels (dùng font_small cache)
        def _lab(txt, x, y):
            screen.blit(font_small.render(txt, True, (20, 20, 20)), (x, y))
        _lab("Board", board_rect.x + 8, board_rect.y + 6)
        _lab("Input", input_rect.x + 8, input_rect.y + 6)
        _lab("Warped", warped_rect.x + 8, warped_rect.y + 6)

        # Right panel
        draw_right_panel(
            screen, font, active_tab, ui,
            replay_moves, replay_index, replay_auto, record_filename,
            camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index,
            right_panel, rect_tabs, rect_dropdowns, rect_buttons, rect_lists, rect_playback,
            processing_frame, last_board_img, [], has_board_preview, clock, rect_board_area.width, board_locked,
            debug_panel, available_cameras,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y,
            ui_flip_view
        )

        # Debug overlay — draw() hiện là NO-OP (không vẽ thông số), nhưng giữ call để tương thích
        if debug_enabled:
            debug_panel.draw(screen, font, {
                "fps": fps_current,
                "grab_every": frame_grab_every_current,
                "piece_conf": piece_conf_current,
                "rotate_steps": rotate_steps,
                "camera_status": camera_status,
                "board_locked": board_locked,
                "has_quad": board_quad is not None,
                "recording": recording,
                "manual_grid_visible": manual_enabled,
                "swap_axes": manual_swap_axes,
                "river_extra": manual_river_extra,
                "scale_x": manual_scale_x,
                "scale_y": manual_scale_y,
                "offset_x": manual_offset_x,
                "offset_y": manual_offset_y,
                "ui_flip_view": ui_flip_view,
                "cam_thread_fps": 10,
                "last_detect_seq": last_detect_seq,
            })

        # ===== Dirty rects update thay vì flip toàn màn hình =====
        update_rects = [rect_top_tabs["bar"], board_rect, input_rect, warped_rect, right_panel]
        pygame.display.update(update_rects)

        clock.tick(fps_current)
        frame_idx += 1

    # cleanup
    if camera_thread is not None:
        camera_thread.stop()
    pygame.quit()
    sys.exit()
