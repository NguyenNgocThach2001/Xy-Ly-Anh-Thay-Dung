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
from pygame_board.pygame_board import PygameBoard, grid_from_pieces, apply_move
from app_state import AppState

from pygame_board.record_panel import (
    handle_record_click,
    draw_record_tab,
    AppState,
    RecordPanelRects,
)

from pygame_board.replay_panel import (
    handle_replay_click,
    get_replay_state,
    draw_replay_tab,
)

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Constants
DEFAULT_FPS = 30
PIECE_CONFIDENCE = 0.7
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
FRAME_GRAB_EVERY = 1
CAMERA_CANDIDATES = [0, 1, 2, 3]
FPS_OPTIONS = [5, 10, 15, 25, 30, 60, 90, 120]
FRAME_GRAB_OPTIONS = [1, 2, 3, 5, 10]
PIECE_CONF_OPTIONS = [0.5, 0.6, 0.7, 0.8]

LEFT_COL_RATIO = 0.58
TOPBAR_H = 40

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

class CameraReader:
    # camera fps
    def __init__(self, index: int, width: int, height: int, target_fps: int = 30):
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.target_fps = max(1, int(target_fps))

        self._cap = None
        self._lock = threading.Lock()

        self._latest_rotated = None
        self._latest_warp = None
        self._latest_grid_info = None
        self._latest_seq = 0

        self._running = False
        self._thread = None

        self.ready = False
        self._last_ok_ts = 0.0
        self._reopen_requested = False

        self._rotate_steps = 0
        self._board_quad = None
        self._board_locked = False

    def set_index(self, index: int):
        index = int(index)
        if index != self.index:
            self.index = index
            self._reopen_requested = True

    def set_rotation(self, steps: int):
        self._rotate_steps = int(steps) % 4

    def set_quad(self, quad):
        self._board_quad = quad

    def set_board_locked(self, locked: bool):
        self._board_locked = bool(locked)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True) # daemon thread, kill theo luong chinh
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

    def get_frame(self):
        with self._lock:
            return None if self._latest_rotated is None else self._latest_rotated.copy()

    def get_warp_and_info(self):
        with self._lock:
            if self._latest_warp is None:
                return None, None
            return self._latest_warp.copy(), dict(self._latest_grid_info or {})

    def get_warp_and_info_seq(self):
        with self._lock:
            if self._latest_warp is None:
                return None, None, self._latest_seq
            return self._latest_warp.copy(), dict(self._latest_grid_info or {}), self._latest_seq

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
        min_sleep = 1.0 / float(self.target_fps)
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
                rs = self._rotate_steps # xoay 
                if rs == 1:
                    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rs == 2:
                    rotated = cv2.rotate(frame, cv2.ROTATE_180)
                elif rs == 3:
                    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                else:
                    rotated = frame

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
                    self._latest_seq += 1

                self._last_ok_ts = now
                self.ready = True
                time.sleep(min_sleep)
            else:
                if now - self._last_ok_ts > 1.0:
                    self.ready = False
                time.sleep(0.05)

def handle_events(event, state: AppState, FPS_OPTIONS, FRAME_GRAB_OPTIONS, PIECE_CONF_OPTIONS):
    if event.type == pygame.QUIT:
        state.running = False
        return

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            state.running = False
        elif event.key == pygame.K_TAB:
            state.active_tab = (state.active_tab + 1) % 2
        elif event.key == pygame.K_r:
            state.rotate_steps = (state.rotate_steps + 1) % 4

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        mx, my = event.pos
        if state.rect_top_tabs["record"].collidepoint(mx, my):
            state.active_tab = 0
        elif state.rect_top_tabs["replay"].collidepoint(mx, my):
            state.active_tab = 1
        if state.rect_tabs["record"].collidepoint(mx, my):
            state.active_tab = 0
        elif state.rect_tabs["replay"].collidepoint(mx, my):
            state.active_tab = 1

        if state.active_tab == 0:
            prev_recording = state.recording
            clicked_record_button = state.rect_buttons["record"].collidepoint(mx, my)
            rects = RecordPanelRects(
                test_piece=state.rect_buttons["testpiece"],
                rotate=state.rect_buttons["rotate"],
                camera=state.rect_buttons["checkcam"],
                segment=state.rect_buttons["segment"],
                record=state.rect_buttons["record"],
                fps_dropdown=state.rect_dropdowns["fps"],
                grab_dropdown=state.rect_dropdowns["grab"],
                camera_dropdown=state.rect_dropdowns["camera"],
                piece_conf_dropdown=state.rect_dropdowns["pconf"],
                manual_toggle=state.rect_buttons.get("manual_toggle"),
                swap_axes=state.rect_buttons.get("swap_axes"),
                river_minus=state.rect_buttons.get("river_minus"),
                river_plus=state.rect_buttons.get("river_plus"),
                scx_minus=state.rect_buttons.get("scx_minus"),
                scx_plus=state.rect_buttons.get("scx_plus"),
                scy_minus=state.rect_buttons.get("scy_minus"),
                scy_plus=state.rect_buttons.get("scy_plus"),
                offx_minus=state.rect_buttons.get("offx_minus"),
                offx_plus=state.rect_buttons.get("offx_plus"),
                offy_minus=state.rect_buttons.get("offy_minus"),
                offy_plus=state.rect_buttons.get("offy_plus"),
            )

        options = {
            "camera_indices": state.available_cameras,
            "fps_options": FPS_OPTIONS,
            "frame_grab_options": FRAME_GRAB_OPTIONS,
            "piece_conf_options": PIECE_CONF_OPTIONS,
        }

        if state.active_tab == 0:
            prev_recording = state.recording
            clicked_record_button = state.rect_buttons["record"].collidepoint(mx, my)
            handle_record_click(mx, my, state, rects, options)
            if clicked_record_button and prev_recording and not state.recording:
                try:
                    state.ui.reset()
                except Exception:
                    pass
            if state.segment_requested == ("__TEST_PIECE__",):
                state.test_piece_once_requested = True
                state.segment_requested = False
            elif state.segment_requested:
                state.recording = False
                state.record_status = "Idle"
            if state.rect_buttons.get("uiflip") and state.rect_buttons["uiflip"].collidepoint(mx, my):
                state.ui_flip_view = not state.ui_flip_view

        elif state.active_tab == 1:
            handle_replay_click(mx, my, state.rect_playback, state)

def _as_int(value, default=0):
        try:
            if isinstance(value, (list, tuple)):
                if not value:
                    return int(default)
                value = value[0]
            return int(value)
        except Exception:
            return int(default)

def detect_and_orient_pieces(board_img, grid_info, piece_model, piece_conf, flip_view):
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
        if key in info and info[key] is not None:
            try:
                q = _order_quad(info[key])
                if q.shape == (4, 2):
                    return q
            except Exception:
                pass
    return None

def pieces_from_grid(grid):
    out = []
    for r, row in enumerate(grid):
        for c, name in enumerate(row):
            if name is not None:
                out.append((name, (c, r)))
    return out

def bgr_to_surface_scaled(bgr_img, dest_w, dest_h):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        surf = pygame.image.frombuffer(rgb.data, (w, h), "RGB")
        if w != dest_w or h != dest_h:
            surf = pygame.transform.smoothscale(surf, (dest_w, dest_h))
        return surf
    


def run():
    pygame.init()
    pygame.font.init()
    screen_info = pygame.display.Info()
    W, H = screen_info.current_w, screen_info.current_h
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Chinese Chess App")

    font = load_app_font(20)
    font_small = load_app_font(16)
    state = AppState()
    
    def _lab(txt, x, y):
        screen.blit(font_small.render(txt, True, (20, 20, 20)), (x, y))
        _lab("Board", board_rect.x + 8, board_rect.y + 6)
        _lab("Input", input_rect.x + 8, input_rect.y + 6)
        _lab("Warped", warped_rect.x + 8, warped_rect.y + 6)
    
    state.rect_top_tabs = { 
        "bar": pygame.Rect(0, 0, W, TOPBAR_H),  # thanh ngang header để chọn tab
        "record": pygame.Rect(10, 6, 120, TOPBAR_H - 12),       #button chọn record
        "replay": pygame.Rect(140, 6, 120, TOPBAR_H - 12),      #button chọn replay
    }
    content_h = H - TOPBAR_H
    left_w = int(W * 0.68) # nửa bên trái, bàn cờ và cam
    right_w = W - left_w # nửa bên phải, panel.
    view_area = pygame.Rect(0, TOPBAR_H, left_w, content_h)  #  vẽ diện tích vùng bàn cờ
    right_panel = pygame.Rect(left_w, TOPBAR_H, right_w, content_h) # vẽ diện tích vùng panel bên phải
    top_h = int(content_h * 0.08) # chiều cao tab ẩn fast click
    tab_bar = pygame.Rect(right_panel.x, right_panel.y, right_panel.width, top_h) # vẽ tab bar
    tab_w = right_w // 2 # chiều rộng tab ẩn
    state.rect_tabs = {
        "bar": tab_bar,
        "record": pygame.Rect(right_panel.x, right_panel.y, tab_w, top_h),
        "replay": pygame.Rect(right_panel.x + 1 * tab_w, right_panel.y, tab_w, top_h),
    }
    dd_w = right_w - 40 # chiều rộng của dropdown
    x0 = right_panel.x + 20 # tọa độ bắt đầu của dropdown x (trái trên là trục tọa độ)
    y0 = right_panel.y + top_h + 20 # tọa độ bắt đầu của drop down y
    dd_h = 30 # chiều cao của dropdown
    state.rect_dropdowns = { # định nghĩa các drop down
        "fps": pygame.Rect(x0, y0, dd_w, dd_h),
        "grab": pygame.Rect(x0, y0 + 50, dd_w, dd_h),
        "camera": pygame.Rect(x0, y0 + 100, dd_w, dd_h),
        "pconf": pygame.Rect(x0, y0 + 150, dd_w, dd_h),
    }
    btn_w, btn_h = (dd_w - 10) // 2, 36 # độ rộng, cao của btn
    btn_y = y0 + 210 # vị trí của btn
    state.rect_buttons = {  # dict các nút điều khiển
        "rotate": pygame.Rect(x0, btn_y, btn_w, btn_h),
        "checkcam": pygame.Rect(x0 + btn_w + 10, btn_y, btn_w, btn_h),
        "segment": pygame.Rect(x0, btn_y + 50, dd_w, btn_h),
        "record": pygame.Rect(x0, btn_y + 100, dd_w, btn_h),
        "testpiece": pygame.Rect(x0, btn_y + 200, dd_w, btn_h),
        "uiflip": pygame.Rect(x0, btn_y + 250, dd_w, btn_h),
    }
    mg_y = btn_y + 300      # move log
    mg_gap = 8 # khoảng cách giữa các move log
    state.rect_buttons["manual_toggle"] = pygame.Rect(x0, mg_y, dd_w, btn_h)            
    state.rect_buttons["swap_axes"] = pygame.Rect(x0, mg_y + (btn_h + mg_gap), dd_w, btn_h)
    state.rect_buttons["river_minus"] = pygame.Rect(x0, mg_y + 2*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["river_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 2*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["scx_minus"] = pygame.Rect(x0, mg_y + 3*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["scx_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 3*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["scy_minus"] = pygame.Rect(x0, mg_y + 4*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["scy_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 4*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["offx_minus"] = pygame.Rect(x0, mg_y + 5*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["offx_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 5*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["offy_minus"] = pygame.Rect(x0, mg_y + 6*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_buttons["offy_plus"]  = pygame.Rect(x0 + (dd_w + 10)//2, mg_y + 6*(btn_h + mg_gap), (dd_w - 10)//2, btn_h)
    state.rect_lists = {"history": pygame.Rect(x0, y0, dd_w, content_h - (y0 - right_panel.y) - 40)} # dict các ván đã record
    state.rect_playback = { # các btn bên panel replay
        "prev": pygame.Rect(x0, y0, (dd_w - 10) // 2, dd_h),
        "next": pygame.Rect(x0 + (dd_w + 10) // 2, y0, (dd_w - 10) // 2, dd_h),
        "auto": pygame.Rect(x0, y0 + 50, dd_w, dd_h),
    }
    state.rect_right_panel = right_panel # dùng chung
    state.rect_board_area = view_area   # dùng chung

    clock = pygame.time.Clock() # tạo 1 đối tượng clock
    state.fps_current = DEFAULT_FPS
    state.frame_grab_every_current = FRAME_GRAB_EVERY
    state.piece_conf_current = PIECE_CONFIDENCE

    board_model = load_board_model() # load model yolo segment
    piece_model = load_piece_model() # load model yolo detection

    state.camera_status = "Checking cameras..."
    for cam_idx in CAMERA_CANDIDATES: # kiem tra camera nao chay duoc
        cap_test = cv2.VideoCapture(cam_idx)
        cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        ok, _ = cap_test.read()
        if ok:
            state.available_cameras.append(cam_idx)
        cap_test.release()

    #chon camera dau tien hop le
    state.selected_camera_index = state.available_cameras[0] if state.available_cameras else 0
    if state.available_cameras:
        state.camera_ready = True
        state.camera_status = f"Found {len(state.available_cameras)} camera(s). Using {state.selected_camera_index}"
    else:
        state.camera_ready = False
        state.camera_status = "No camera found"

    # neu co cam hop le thi bat dau thread cam
    camera_thread = None
    if state.camera_ready:
        camera_thread = CameraReader(state.selected_camera_index, CAPTURE_WIDTH, CAPTURE_HEIGHT, target_fps=10)
        camera_thread.set_rotation(state.rotate_steps)
        camera_thread.start()

    board_locked = False  # khoa ban co
    board_quad = None   # ban co da detect
    last_board_img = None   # 
    last_grid_info = None   # 

    # khởi tạo UI bàn cờ
    state.ui = PygameBoard(draw_board, draw_pieces, font, init_full=False)
    state.debug_enabled = True

    # tạo folder replay nếu chư có
    os.makedirs("plays", exist_ok=True)
    frame_idx = 0
    last_grabbed = -999
    input_preview_surface = None
    warped_preview_surface = None
    test_piece_overlay = None

    last_detect_seq = -1

    


    while state.running:
        # handle event
        for event in pygame.event.get():
            handle_events(event, state, FPS_OPTIONS, FRAME_GRAB_OPTIONS, PIECE_CONF_OPTIONS)
        state.rotate_steps = _as_int(state.rotate_steps, 0)
        state.ui.rotate_steps = state.rotate_steps

        # set camera
        if state.camera_ready and camera_thread is not None:
            if camera_thread.index != state.selected_camera_index:
                camera_thread.set_index(state.selected_camera_index)
            camera_thread.set_rotation(state.rotate_steps)
            camera_thread.set_board_locked(board_locked)
            camera_thread.set_quad(board_quad)

        #get input
        processing_frame = None
        if camera_thread is not None and (frame_idx - last_grabbed) >= state.frame_grab_every_current:
            pf = camera_thread.get_frame()
            if pf is not None:
                processing_frame = pf
                last_grabbed = frame_idx

        # SEGMENT
        if state.segment_requested and processing_frame is not None:
            board_img, found, info = detect_board_with_mode(processing_frame, mode="auto", model=board_model)
            if found:
                q = _get_quad_from_info(info)
                if q is not None:
                    board_quad = q
                    last_board_img, last_grid_info = camera_thread.get_warp_and_info()
                else:
                    board_quad = None
                    last_board_img = None
                    last_grid_info = None
                board_locked = True
                state.camera_status = "Board segmented and locked."
            else:
                state.camera_status = "Segment failed. Try again."
                board_locked = False
                board_quad = None
                last_board_img = None
                last_grid_info = None
            state.segment_requested = False

        if board_locked and camera_thread is not None:
            wb, gi = camera_thread.get_warp_and_info()
            if wb is not None:
                last_board_img = wb
                last_grid_info = gi

        # ONE-SHOT TEST PIECE
        if state.test_piece_once_requested and last_board_img is not None and last_grid_info is not None:
            try:
                gi = dict(last_grid_info or {})
                gi["swap_axes"] = bool(state.manual_swap_axes)
                gi["river_extra"] = float(state.manual_river_extra)
                gi["scale_x"] = float(state.manual_scale_x)
                gi["scale_y"] = float(state.manual_scale_y)
                gi["offset_x"] = float(state.manual_offset_x)
                gi["offset_y"] = float(state.manual_offset_y)
                test_pieces, _ = detect_and_orient_pieces(
                    last_board_img, gi, piece_model, state.piece_conf_current, state.ui_flip_view
                )
                test_piece_overlay = test_pieces or []
            except Exception:
                test_piece_overlay = []
            finally:
                state.test_piece_once_requested = False

        # RECORD PIPELINE
        if state.active_tab == 0 and state.recording and board_locked and camera_thread is not None:
            try:
                warped_img, grid_info, seq = camera_thread.get_warp_and_info_seq()
                if warped_img is not None and grid_info is not None:
                    if seq != last_detect_seq:
                        last_detect_seq = seq
                        gi = dict(grid_info)
                        gi["swap_axes"] = bool(state.manual_swap_axes)
                        gi["river_extra"] = float(state.manual_river_extra)
                        gi["scale_x"] = float(state.manual_scale_x)
                        gi["scale_y"] = float(state.manual_scale_y)
                        gi["offset_x"] = float(state.manual_offset_x)
                        gi["offset_y"] = float(state.manual_offset_y)
                        pieces, _ = detect_and_orient_pieces(
                            warped_img, gi, piece_model, state.piece_conf_current, state.ui_flip_view
                        )
                        state.ui.update_state(True, pieces)
                        if state.ui.has_new_move():
                            move_text = state.ui.get_last_move_text()
                            if move_text:
                                if not state.record_filename:
                                    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    state.record_filename = f"plays/play_{dt}.txt"
                                with open(state.record_filename, "a", encoding="utf-8") as f:
                                    f.write(move_text + "\n")
            except Exception:
                pass

        if state.recording and test_piece_overlay:
            test_piece_overlay = None

        screen.fill((20, 20, 20))

        # Top bar
        pygame.draw.rect(screen, (25, 25, 25), state.rect_top_tabs["bar"])
        for key, label in [("record", "Record"), ("replay", "Replay")]:
            r = state.rect_top_tabs[key]
            color = (60, 60, 60)
            if r.collidepoint(pygame.mouse.get_pos()):
                color = (80, 80, 80)
            if (state.active_tab == 0 and key == "record") or (state.active_tab == 1 and key == "replay"):
                color = (95, 95, 95)
            pygame.draw.rect(screen, color, r, border_radius=6)
            pygame.draw.rect(screen, (120, 120, 120), r, 1, border_radius=6)
            text = font.render(label, True, (255, 255, 255))
            screen.blit(text, (r.x + 10, r.y + 6))

        pygame.draw.rect(screen, (50, 50, 50), state.rect_board_area)
        margin = 24
        inner = state.rect_board_area.inflate(-2 * margin, -2 * margin)
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

        board_surface = pygame.Surface((board_rect.width, board_rect.height)).convert()
        board_surface.fill((235, 235, 210))
        draw_board(board_surface)

        if state.active_tab == 1:
            moves, step, _auto = get_replay_state(state)
            start_pieces = state.ui._starting_position()
            grid = grid_from_pieces(start_pieces)
            for i in range(min(step, len(moves))):
                (sr, sc), (dr, dc) = moves[i]
                grid = apply_move(grid, (sr, sc), (dr, dc))
            render_state = pieces_from_grid(grid)
        else:
            render_state = state.ui.state_live

        draw_pieces(board_surface, font, render_state)
        if (not state.recording) and (state.active_tab != 1) and test_piece_overlay:
            overlay = test_piece_overlay
            draw_pieces(board_surface, font, overlay)

        screen.blit(board_surface, (board_rect.x, board_rect.y))

        

        if processing_frame is not None:
            input_preview_surface = bgr_to_surface_scaled(processing_frame, input_rect.width, input_rect.height).convert()
        if input_preview_surface is not None:
            screen.blit(input_preview_surface, (input_rect.x, input_rect.y))
        if last_board_img is not None:
            warped_preview_surface = bgr_to_surface_scaled(last_board_img, warped_rect.width, warped_rect.height).convert()
        if warped_preview_surface is not None:
            screen.blit(warped_preview_surface, (warped_rect.x, warped_rect.y))
            if last_grid_info is not None:
                man = {
                    "visible": state.manual_enabled,
                    "swap_axes": state.manual_swap_axes,
                    "river_extra": state.manual_river_extra,
                    "scale_x": state.manual_scale_x,
                    "scale_y": state.manual_scale_y,
                    "offset_x": state.manual_offset_x,
                    "offset_y": state.manual_offset_y,
                }
                overlay_surface = get_grid_overlay(last_grid_info, man, (warped_rect.width, warped_rect.height))
                if overlay_surface is not None:
                    screen.blit(overlay_surface, (warped_rect.x, warped_rect.y))
                    
        
        if state.active_tab == 0:
            draw_record_tab(
                screen,
                test_piece_button_rect=state.rect_buttons["testpiece"],
                panel_origin_y=state.rect_tabs["bar"].bottom,
                mono=font,
                rotate_button_rect=state.rect_buttons["rotate"],
                camera_button_rect=state.rect_buttons["checkcam"],
                segment_button_rect=state.rect_buttons["segment"],
                record_button_rect=state.rect_buttons["record"],
                fps_dropdown_rect=state.rect_dropdowns["fps"],
                grab_dropdown_rect=state.rect_dropdowns["grab"],
                camera_dropdown_rect=state.rect_dropdowns["camera"],
                piece_conf_dropdown_rect=state.rect_dropdowns["pconf"],
                camera_indices=state.available_cameras,
                camera_ready=state.camera_ready,
                recording=state.recording,
                fps_dropdown_open=state.fps_dropdown_open,
                fps_current=state.fps_current,
                frame_grab_dropdown_open=state.frame_grab_dropdown_open,
                frame_grab_every_current=state.frame_grab_every_current,
                camera_dropdown_open=state.camera_dropdown_open,
                selected_camera_index=state.selected_camera_index,
                piece_conf_dropdown_open=state.piece_conf_dropdown_open,
                piece_conf_current=state.piece_conf_current,
                camera_status=state.camera_status,
                record_status=state.record_status,
                fps_options=FPS_OPTIONS,
                frame_grab_options=FRAME_GRAB_OPTIONS,
                piece_conf_options=PIECE_CONF_OPTIONS,
                rotate_steps=state.rotate_steps,
                board_locked=board_locked,
                manual_toggle_rect=state.rect_buttons.get("manual_toggle"),
                swap_axes_rect=state.rect_buttons.get("swap_axes"),
                river_minus_rect=state.rect_buttons.get("river_minus"),
                river_plus_rect=state.rect_buttons.get("river_plus"),
                scx_minus_rect=state.rect_buttons.get("scx_minus"),
                scx_plus_rect=state.rect_buttons.get("scx_plus"),
                scy_minus_rect=state.rect_buttons.get("scy_minus"),
                scy_plus_rect=state.rect_buttons.get("scy_plus"),
                offx_minus_rect=state.rect_buttons.get("offx_minus"),
                offx_plus_rect=state.rect_buttons.get("offx_plus"),
                offy_minus_rect=state.rect_buttons.get("offy_minus"),
                offy_plus_rect=state.rect_buttons.get("offy_plus"),
                manual_enabled=state.manual_enabled,
                manual_swap_axes=state.manual_swap_axes,
                manual_river_extra=state.manual_river_extra,
                manual_scale_x=state.manual_scale_x,
                manual_scale_y=state.manual_scale_y,
                manual_offset_x=state.manual_offset_x,
                manual_offset_y=state.manual_offset_y,
                record_filename=state.record_filename,
                right_panel_rect=state.rect_right_panel,
            )
        else:
            draw_replay_tab(screen, font, state, state.rect_playback)
        pygame.display.update([state.rect_top_tabs["bar"], board_rect, input_rect, warped_rect, state.rect_right_panel])
        clock.tick(state.fps_current)
        frame_idx += 1

    if camera_thread is not None:
        camera_thread.stop()
    pygame.quit()
    sys.exit()

def get_grid_overlay(grid_info, manual_params, size_tuple):
    
    if not manual_params.get("visible", False):
        return None

    w, h = size_tuple
    # plane voi alpha, muc dich de đè lên warp
    surf = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()

    # Vẽ lưới manual lên surf 
    sx = w / 640.0
    sy = h / 640.0

    pad_x = float(grid_info.get("pad_x", 0)) * sx
    pad_y = float(grid_info.get("pad_y", 0)) * sy
    usable_w = float(grid_info.get("usable_w", w)) * sx
    usable_h = float(grid_info.get("usable_h", h)) * sy

    river_extra = max(0.0, float(manual_params.get("river_extra", 0.0)))
    scale_x     = float(manual_params.get("scale_x", 1.0))
    scale_y     = float(manual_params.get("scale_y", 1.0))
    off_x       = float(manual_params.get("offset_x", 0.0)) * sx
    off_y       = float(manual_params.get("offset_y", 0.0)) * sy

    base_w = usable_w * max(0.2, min(5.0, scale_x))
    base_h = usable_h * max(0.2, min(5.0, scale_y))

    x0 = pad_x + off_x
    y0 = pad_y + off_y

    GRID = (40, 40, 40, 170)
    RIVER = (220, 50, 50, 200)
    OUTLINE = (10, 10, 10, 200)

    cell_x = base_w / 8.0
    cell_y = base_h / 9.0
    y_river_top = y0 + 4.0 * cell_y
    y_river_bot = y_river_top + river_extra
    y_bottom = y0 + 8.0 * cell_y + river_extra

    pygame.draw.rect(
        surf, OUTLINE,
        pygame.Rect(int(x0), int(y0), int(base_w), int(y_bottom - y0)), 2
    )
    # Vertical
    for i in range(9):
        x = int(round(x0 + i * cell_x))
        pygame.draw.line(surf, GRID, (x, int(y0)), (x, int(y_bottom)), 1)
    # Horizontal trước/sau sông
    for r in range(5):
        y = int(round(y0 + r * cell_y))
        pygame.draw.line(surf, GRID, (int(x0), y), (int(x0 + base_w), y), 1)
    for r in range(6, 10):
        y = int(round(y_river_bot + (r - 5) * cell_y))
        pygame.draw.line(surf, GRID, (int(x0), y), (int(x0 + base_w), y), 1)
    # River lines
    pygame.draw.line(surf, RIVER, (int(x0), int(round(y_river_top))),
        (int(x0 + base_w), int(round(y_river_top))), 2)
    pygame.draw.line(surf, RIVER, (int(x0), int(round(y_river_bot))),
        (int(x0 + base_w), int(round(y_river_bot))), 2)
    

    return surf
