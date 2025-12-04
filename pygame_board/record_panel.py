import os
import cv2
import pygame
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from app_state import AppState

@dataclass
class RecordPanelRects:
    test_piece: Any
    rotate: Any
    camera: Any
    segment: Any
    record: Any
    fps_dropdown: Any
    grab_dropdown: Any
    camera_dropdown: Any
    piece_conf_dropdown: Any
    manual_toggle: Optional[Any] = None
    swap_axes: Optional[Any] = None
    river_minus: Optional[Any] = None
    river_plus: Optional[Any] = None
    scx_minus: Optional[Any] = None
    scx_plus: Optional[Any] = None
    scy_minus: Optional[Any] = None
    scy_plus: Optional[Any] = None
    offx_minus: Optional[Any] = None
    offx_plus: Optional[Any] = None
    offy_minus: Optional[Any] = None
    offy_plus: Optional[Any] = None

def handle_step_button(mx, my, rect, state, key, step, minval, maxval):
    if rect and rect.collidepoint(mx, my):
        current = getattr(state, key)
        setattr(state, key, max(minval, min(maxval, current + step)))
        return True
    return False

def handle_record_click(
    mx,
    my,
    state,
    rects: RecordPanelRects,
    options: dict
):
    
    option_h = 28
    if state.fps_dropdown_open:
        for i, val in enumerate(options["fps_options"]):
            opt_rect = pygame.Rect(
                rects.fps_dropdown.x,
                rects.fps_dropdown.y + (i + 1) * option_h,
                rects.fps_dropdown.width,
                option_h,
            )
            if opt_rect.collidepoint(mx, my):
                state.fps_current = val
                state.fps_dropdown_open = False
                return
    if state.frame_grab_dropdown_open:
        for i, val in enumerate(options["frame_grab_options"]):
            opt_rect = pygame.Rect(
                rects.grab_dropdown.x,
                rects.grab_dropdown.y + (i + 1) * option_h,
                rects.grab_dropdown.width,
                option_h,
            )
            if opt_rect.collidepoint(mx, my):
                state.frame_grab_every_current = val
                state.frame_grab_dropdown_open = False
                return
    if state.camera_dropdown_open:
        for i, cam_idx in enumerate(options["camera_indices"]):
            opt_rect = pygame.Rect(
                rects.camera_dropdown.x,
                rects.camera_dropdown.y + (i + 1) * option_h,
                rects.camera_dropdown.width,
                option_h,
            )
            if opt_rect.collidepoint(mx, my):
                state.selected_camera_index = cam_idx
                state.camera_dropdown_open = False
                return
    if state.piece_conf_dropdown_open:
        for i, val in enumerate(options["piece_conf_options"]):
            opt_rect = pygame.Rect(
                rects.piece_conf_dropdown.x,
                rects.piece_conf_dropdown.y + (i + 1) * option_h,
                rects.piece_conf_dropdown.width,
                option_h,
            )
            if opt_rect.collidepoint(mx, my):
                state.piece_conf_current = val
                state.piece_conf_dropdown_open = False
                return

    if rects.fps_dropdown.collidepoint(mx, my):
        state.fps_dropdown_open = not state.fps_dropdown_open
        state.frame_grab_dropdown_open = False
        state.camera_dropdown_open = False
        state.piece_conf_dropdown_open = False
        return
    elif rects.grab_dropdown.collidepoint(mx, my):
        state.frame_grab_dropdown_open = not state.frame_grab_dropdown_open
        state.fps_dropdown_open = False
        state.camera_dropdown_open = False
        state.piece_conf_dropdown_open = False
        return
    elif rects.camera_dropdown.collidepoint(mx, my):
        state.camera_dropdown_open = not state.camera_dropdown_open
        state.fps_dropdown_open = False
        state.frame_grab_dropdown_open = False
        state.piece_conf_dropdown_open = False
        return
    elif rects.piece_conf_dropdown.collidepoint(mx, my):
        state.piece_conf_dropdown_open = not state.piece_conf_dropdown_open
        state.fps_dropdown_open = False
        state.frame_grab_dropdown_open = False
        state.camera_dropdown_open = False
        return
    elif rects.rotate.collidepoint(mx, my):
        state.rotate_steps = (state.rotate_steps + 1) % 4
        return

    if rects.test_piece.collidepoint(mx, my):
        state.segment_requested = ("__TEST_PIECE__",)
        return

    elif rects.camera.collidepoint(mx, my):
        cap = cv2.VideoCapture(state.selected_camera_index)
        ok, _ = cap.read()
        if ok:
            state.camera_ready = True
            state.camera_status = f"Camera {state.selected_camera_index} OK"
        else:
            state.camera_ready = False
            state.camera_status = "Cannot open camera."
        cap.release()
        return

    elif rects.segment.collidepoint(mx, my):
        state.segment_requested = True
        state.recording = False
        state.record_status = "Idle"
        return

    elif rects.record.collidepoint(mx, my):
        if not state.recording:
            if not state.camera_ready:
                state.record_status = "Camera not ready."
                return
            state.recording = True
            state.record_status = "Recording..."
        else:
            state.recording = False
            state.record_status = "Stopped."
        return
    
    toggles = [
        (rects.manual_toggle, "manual_enabled"),
        (rects.swap_axes, "manual_swap_axes")
    ]
    for rect, key in toggles:
        if rect and rect.collidepoint(mx, my):
            setattr(state, key, not getattr(state, key))
            return

    manual_controls = [
        (rects.river_minus, "manual_river_extra", -2.0, 0.0, 200.0),
        (rects.river_plus,  "manual_river_extra", +2.0, 0.0, 200.0),
        (rects.scx_minus,   "manual_scale_x",     -0.02, 0.5, 1.5),
        (rects.scx_plus,    "manual_scale_x",     +0.02, 0.5, 1.5),
        (rects.scy_minus,   "manual_scale_y",     -0.02, 0.5, 1.5),
        (rects.scy_plus,    "manual_scale_y",     +0.02, 0.5, 1.5),
        (rects.offx_minus,  "manual_offset_x",    -2.0, -320.0, 320.0),
        (rects.offx_plus,   "manual_offset_x",    +2.0, -320.0, 320.0),
        (rects.offy_minus,  "manual_offset_y",    -2.0, -320.0, 320.0),
        (rects.offy_plus,   "manual_offset_y",    +2.0, -320.0, 320.0),
    ]
    
    for rect, key, step, minval, maxval in manual_controls:
        if handle_step_button(mx, my, rect, state, key, step, minval, maxval):
            return

    state.fps_dropdown_open = False
    state.frame_grab_dropdown_open = False
    state.camera_dropdown_open = False
    state.piece_conf_dropdown_open = False

def draw_record_tab(
    screen,
    test_piece_button_rect,
    panel_origin_y,
    mono,
    rotate_button_rect,
    camera_button_rect,
    segment_button_rect,
    record_button_rect,
    fps_dropdown_rect,
    grab_dropdown_rect,
    camera_dropdown_rect,
    piece_conf_dropdown_rect,
    camera_indices,
    camera_ready,
    recording,
    fps_dropdown_open,
    fps_current,
    frame_grab_dropdown_open,
    frame_grab_every_current,
    camera_dropdown_open,
    selected_camera_index,
    piece_conf_dropdown_open,
    piece_conf_current,
    camera_status,
    record_status,
    fps_options,
    frame_grab_options,
    piece_conf_options,
    rotate_steps,
    board_locked,
    manual_toggle_rect=None,
    swap_axes_rect=None,
    river_minus_rect=None,
    river_plus_rect=None,
    scx_minus_rect=None,
    scx_plus_rect=None,
    scy_minus_rect=None,
    scy_plus_rect=None,
    offx_minus_rect=None,
    offx_plus_rect=None,
    offy_minus_rect=None,
    offy_plus_rect=None,
    manual_enabled=False,
    manual_swap_axes=False,
    manual_river_extra=0.0,
    manual_scale_x=1.0,
    manual_scale_y=1.0,
    manual_offset_x=0.0,
    manual_offset_y=0.0,
    record_filename="",
    right_panel_rect=None,
):
    mouse_pos = pygame.mouse.get_pos()

    def _draw_button(screen, mono, rect, text, mouse_pos, active=False):
        color = (60, 60, 60)
        if rect.collidepoint(mouse_pos):
            color = (80, 80, 80)
        pygame.draw.rect(screen, color, rect, border_radius=6)
        pygame.draw.rect(screen, (120, 120, 120), rect, 1, border_radius=6)
        txt = mono.render(text, True, (255, 255, 255))
        screen.blit(txt, (rect.x + 8, rect.y + 6))

    def _draw_dropdown(screen, mono, rect, text):
        pygame.draw.rect(screen, (50, 50, 50), rect, border_radius=6)
        pygame.draw.rect(screen, (120, 120, 120), rect, 1, border_radius=6)
        txt = mono.render(text, True, (255, 255, 255))
        screen.blit(txt, (rect.x + 8, rect.y + 6))

    def _draw_value_in_right_btn(right_rect, mono, text):
        val = mono.render(text, True, (200, 200, 200))
        vx = right_rect.right - val.get_width() - 8
        vy = right_rect.bottom - val.get_height() - 6
        screen.blit(val, (vx, vy))
    option_h = 28

    _draw_dropdown(screen, mono, fps_dropdown_rect, f"FPS: {fps_current}")
    _draw_dropdown(screen, mono, grab_dropdown_rect, f"Grab every: {frame_grab_every_current}")
    _draw_dropdown(screen, mono, camera_dropdown_rect, f"Camera: {selected_camera_index}")
    _draw_dropdown(screen, mono, piece_conf_dropdown_rect, f"Piece conf: {piece_conf_current:.1f}")

    _draw_button(screen, mono, camera_button_rect, "Check camera", mouse_pos, active=camera_ready)
    _draw_button(screen, mono, segment_button_rect, "Segment board", mouse_pos, active=board_locked)
    _draw_button(screen, mono, rotate_button_rect, f"Rotate 90° (x{rotate_steps})", mouse_pos, active=(rotate_steps % 4 != 0))
    _draw_button(screen, mono, record_button_rect, "Stop" if recording else "Record", mouse_pos, active=recording)
    _draw_button(screen, mono, test_piece_button_rect, "Test piece", mouse_pos, active=False)

    # ==== Manual Grid Section ====
    if manual_toggle_rect is not None:
        _draw_button(screen, mono, manual_toggle_rect, f"Grid Overlay: {'SHOW' if manual_enabled else 'HIDE'}", mouse_pos, active=manual_enabled)
    if swap_axes_rect is not None:
        _draw_button(screen, mono, swap_axes_rect, f"Swap Axes: {'Y→9' if manual_swap_axes else 'X→9'}", mouse_pos, active=True)

    if river_minus_rect and river_plus_rect:
        _draw_button(screen, mono, river_minus_rect, "River -", mouse_pos, active=True)
        _draw_button(screen, mono, river_plus_rect,  "River +", mouse_pos, active=True)
        _draw_value_in_right_btn(river_plus_rect, mono, f"river_extra={manual_river_extra:.0f}px")

    if scx_minus_rect and scx_plus_rect:
        _draw_button(screen, mono, scx_minus_rect, "ScaleX -", mouse_pos, active=True)
        _draw_button(screen, mono, scx_plus_rect,  "ScaleX +", mouse_pos, active=True)
        _draw_value_in_right_btn(scx_plus_rect, mono, f"scale_x={manual_scale_x:.2f}")

    if scy_minus_rect and scy_plus_rect:
        _draw_button(screen, mono, scy_minus_rect, "ScaleY -", mouse_pos, active=True)
        _draw_button(screen, mono, scy_plus_rect,  "ScaleY +", mouse_pos, active=True)
        _draw_value_in_right_btn(scy_plus_rect, mono, f"scale_y={manual_scale_y:.2f}")

    if offx_minus_rect and offx_plus_rect:
        _draw_button(screen, mono, offx_minus_rect, "OffsetX -", mouse_pos, active=True)
        _draw_button(screen, mono, offx_plus_rect,  "OffsetX +", mouse_pos, active=True)
        _draw_value_in_right_btn(offx_plus_rect, mono, f"offset_x={manual_offset_x:.0f}px")

    if offy_minus_rect and offy_plus_rect:
        _draw_button(screen, mono, offy_minus_rect, "OffsetY -", mouse_pos, active=True)
        _draw_button(screen, mono, offy_plus_rect,  "OffsetY +", mouse_pos, active=True)
        _draw_value_in_right_btn(offy_plus_rect, mono, f"offset_y={manual_offset_y:.0f}px")

    if fps_dropdown_open:
        for i, val in enumerate(fps_options):
            opt_rect = pygame.Rect(
                fps_dropdown_rect.x,
                fps_dropdown_rect.y + (i + 1) * option_h,
                fps_dropdown_rect.width,
                option_h,
            )
            pygame.draw.rect(screen, (60, 60, 60), opt_rect, border_radius=4)
            txt = mono.render(str(val), True, (255, 255, 255))
            screen.blit(txt, (opt_rect.x + 8, opt_rect.y + 6))
    if frame_grab_dropdown_open:
        for i, val in enumerate(frame_grab_options):
            opt_rect = pygame.Rect(
                grab_dropdown_rect.x,
                grab_dropdown_rect.y + (i + 1) * option_h,
                grab_dropdown_rect.width,
                option_h,
            )
            pygame.draw.rect(screen, (60, 60, 60), opt_rect, border_radius=4)
            txt = mono.render(str(val), True, (255, 255, 255))
            screen.blit(txt, (opt_rect.x + 8, opt_rect.y + 6))
    if camera_dropdown_open:
        for i, cam_idx in enumerate(camera_indices):
            opt_rect = pygame.Rect(
                camera_dropdown_rect.x,
                camera_dropdown_rect.y + (i + 1) * option_h,
                camera_dropdown_rect.width,
                option_h,
            )
            pygame.draw.rect(screen, (60, 60, 60), opt_rect, border_radius=4)
            txt = mono.render(str(cam_idx), True, (255, 255, 255))
            screen.blit(txt, (opt_rect.x + 8, opt_rect.y + 6))
    if piece_conf_dropdown_open:
        for i, val in enumerate(piece_conf_options):
            opt_rect = pygame.Rect(
                piece_conf_dropdown_rect.x,
                piece_conf_dropdown_rect.y + (i + 1) * option_h,
                piece_conf_dropdown_rect.width,
                option_h,
            )
            pygame.draw.rect(screen, (60, 60, 60), opt_rect, border_radius=4)
            txt = mono.render(f"{val:.1f}", True, (255, 255, 255))
            screen.blit(txt, (opt_rect.x + 8, opt_rect.y + 6))

    bottoms = []
    for r in [
        offy_plus_rect, offx_plus_rect, scy_plus_rect, scx_plus_rect,
        river_plus_rect, swap_axes_rect, manual_toggle_rect,
    ]:
        if r is not None:
            bottoms.append(r.bottom)
    y_start = (max(bottoms) + 14) if bottoms else (panel_origin_y + 14)
    if right_panel_rect is not None:
        x = fps_dropdown_rect.x
        w = fps_dropdown_rect.width
        y = max(y_start, right_panel_rect.y + 10)
        h = max(0, right_panel_rect.bottom - y - 12)
    else:
        x = fps_dropdown_rect.x
        w = fps_dropdown_rect.width
        y = y_start
        h = 180

    if h > 24:
        box_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, (28, 28, 28), box_rect, border_radius=6)
        pygame.draw.rect(screen, (90, 90, 90), box_rect, 1, border_radius=6)
        title = mono.render("Move Log", True, (210, 210, 210))
        screen.blit(title, (x + 8, y + 6))
        inner_x = x + 8
        inner_y = y + 28
        inner_w = w - 16
        inner_h = h - 36
        lines = []
        if record_filename and os.path.isfile(record_filename):
            try:
                with open(record_filename, "r", encoding="utf-8") as f:
                    lines = [ln.rstrip("\n") for ln in f.readlines()]
            except Exception:
                lines = ["(Error reading current play file)"]
        else:
            if record_filename:
                lines = [f"(Waiting for moves in: {os.path.basename(record_filename)})"]
            else:
                lines = ["(Press 'Record' and make a move to start logging)"]
        line_h = mono.get_linesize()
        max_lines = max(1, inner_h // line_h)
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        cy = inner_y
        for s in lines:
            surf = mono.render(s, True, (235, 235, 235))
            screen.blit(surf, (inner_x, cy))
            cy += line_h