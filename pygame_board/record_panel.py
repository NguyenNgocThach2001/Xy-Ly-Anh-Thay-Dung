import os
import cv2
import pygame

# ==============================
# DebugPanel (đã gộp vào record_panel)
# - Chỉ hiển thị panel ảnh: Raw / Warp / Overlay.
# - draw(...) là NO-OP để không vẽ thông số text.
# ==============================
class DebugPanel:
    """Hiển thị warp & overlay trực tiếp trong Pygame (không lưu file)."""
    def __init__(self, screen, origin=(820, 20), panel_w=280, gap=16, font=None):
        self.screen = screen
        self.ox, self.oy = origin
        self.panel_w = panel_w
        self.gap = gap
        self.font = font or pygame.font.SysFont("consolas", 18)

    @staticmethod
    def cv_to_surface(img_bgr, max_w):
        if img_bgr is None:
            return None
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(1.0, max_w / float(w)) if w > 0 else 1.0
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        surface = pygame.image.frombuffer(rgb.tobytes(), (new_w, new_h), "RGB")
        return surface.convert()

    @staticmethod
    def draw_pieces_on_board(board_img, pieces):
        if board_img is None:
            return None
        img = board_img.copy()
        h, w = img.shape[:2]
        if w == 0 or h == 0:
            return img
        cell_w = w / 9.0
        cell_h = h / 10.0
        for label, (col, row) in (pieces or []):
            try:
                cx = int(col * cell_w + cell_w / 2)
                cy = int(row * cell_h + cell_h / 2)
                cv2.circle(img, (cx, cy), 16, (0, 0, 255), 2)
                cv2.putText(
                    img,
                    str(label),
                    (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                continue
        return img

    def _title(self, text, y):
        label = self.font.render(text, True, (20, 20, 20))
        self.screen.blit(label, (self.ox, y))
        return y + label.get_height() + 6

    def show(self, raw_bgr, warp_bgr, pieces, pieces_count, has_board, fps_text=""):
        """Vẽ panel debug ảnh (không lưu file)."""
        y = self.oy
        status = "NO_BOARD" if not has_board else f"BOARD_OK: {pieces_count} pcs"
        header = f"[Debug] {status}"
        if fps_text:
            header += f"  {fps_text}"
        y = self._title(header, y)

        # --- RAW ---
        y = self._title("Raw", y)
        surf_raw = self.cv_to_surface(raw_bgr, self.panel_w)
        if surf_raw:
            self.screen.blit(surf_raw, (self.ox, y))
            y += surf_raw.get_height() + self.gap
        else:
            y = self._title("(no raw)", y)

        # --- WARP ---
        y = self._title("Warp", y)
        surf_warp = self.cv_to_surface(warp_bgr, self.panel_w)
        if surf_warp:
            self.screen.blit(surf_warp, (self.ox, y))
            y += surf_warp.get_height() + self.gap
        else:
            y = self._title("(no warp)", y)

        # --- OVERLAY ---
        y = self._title("Overlay", y)
        overlay = self.draw_pieces_on_board(warp_bgr, pieces) if has_board else None
        surf_overlay = self.cv_to_surface(overlay, self.panel_w)
        if surf_overlay:
            self.screen.blit(surf_overlay, (self.ox, y))
            y += surf_overlay.get_height() + self.gap
        else:
            y = self._title("(no overlay)", y)

    # NO-OP: bỏ hiển thị thông số text
    def draw(self, screen, font, info_dict):
        return

    def reset(self):
        pass


def handle_record_click(
    mx, my,
    test_piece_button_rect,
    rotate_button_rect,
    camera_button_rect,
    segment_button_rect,
    record_button_rect,
    fps_dropdown_rect,
    grab_dropdown_rect,
    camera_dropdown_rect,
    piece_conf_dropdown_rect,
    debug_button_rect,
    camera_indices,
    cap,
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
    fps_options,
    frame_grab_options,
    piece_conf_options,
    # ===== Manual Grid UI (mới) =====
    manual_toggle_rect=None,
    swap_axes_rect=None,
    river_minus_rect=None,
    river_plus_rect=None,
    scx_minus_rect=None,
    scx_plus_rect=None,
    scy_minus_rect=None,
    scy_plus_rect=None,
    # NEW offset controls
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
):
    """Xử lý click ở tab Record; trả về state + cờ segment_requested + các tham số manual-grid."""
    # Click vào dropdown: FPS
    option_h = 28
    if fps_dropdown_open:
        for i, val in enumerate(fps_options):
            opt_rect = pygame.Rect(
                fps_dropdown_rect.x,
                fps_dropdown_rect.y + (i + 1) * option_h,
                fps_dropdown_rect.width,
                option_h,
            )
            if opt_rect.collidepoint(mx, my):
                fps_current = val
                fps_dropdown_open = False
                return (
                    cap, camera_ready, camera_status, recording, record_status,
                    fps_dropdown_open, fps_current,
                    frame_grab_dropdown_open, frame_grab_every_current,
                    camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
                    debug_enabled, selected_camera_index, rotate_steps,
                    False,
                    manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
                    manual_offset_x, manual_offset_y
                )

    # Click vào dropdown: Grab
    if frame_grab_dropdown_open:
        for i, val in enumerate(frame_grab_options):
            opt_rect = pygame.Rect(
                grab_dropdown_rect.x,
                grab_dropdown_rect.y + (i + 1) * option_h,
                grab_dropdown_rect.width,
                option_h,
            )
            if opt_rect.collidepoint(mx, my):
                frame_grab_every_current = val
                frame_grab_dropdown_open = False
                return (
                    cap, camera_ready, camera_status, recording, record_status,
                    fps_dropdown_open, fps_current,
                    frame_grab_dropdown_open, frame_grab_every_current,
                    camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
                    debug_enabled, selected_camera_index, rotate_steps,
                    False,
                    manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
                    manual_offset_x, manual_offset_y
                )

    # Click vào dropdown: Camera
    if camera_dropdown_open:
        for i, cam_idx in enumerate(camera_indices):
            opt_rect = pygame.Rect(
                camera_dropdown_rect.x,
                camera_dropdown_rect.y + (i + 1) * option_h,
                camera_dropdown_rect.width,
                option_h,
            )
            if opt_rect.collidepoint(mx, my):
                selected_camera_index = cam_idx
                camera_dropdown_open = False
                return (
                    cap, camera_ready, camera_status, recording, record_status,
                    fps_dropdown_open, fps_current,
                    frame_grab_dropdown_open, frame_grab_every_current,
                    camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
                    debug_enabled, selected_camera_index, rotate_steps,
                    False,
                    manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
                    manual_offset_x, manual_offset_y
                )

    # Click vào dropdown: Piece conf
    if piece_conf_dropdown_open:
        for i, val in enumerate(piece_conf_options):
            opt_rect = pygame.Rect(
                piece_conf_dropdown_rect.x,
                piece_conf_dropdown_rect.y + (i + 1) * option_h,
                piece_conf_dropdown_rect.width,
                option_h,
            )
            if opt_rect.collidepoint(mx, my):
                piece_conf_current = val
                piece_conf_dropdown_open = False
                return (
                    cap, camera_ready, camera_status, recording, record_status,
                    fps_dropdown_open, fps_current,
                    frame_grab_dropdown_open, frame_grab_every_current,
                    camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
                    debug_enabled, selected_camera_index, rotate_steps,
                    False,
                    manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
                    manual_offset_x, manual_offset_y
                )

    # Nút FPS dropdown
    if fps_dropdown_rect.collidepoint(mx, my):
        fps_dropdown_open = not fps_dropdown_open
        frame_grab_dropdown_open = False
        camera_dropdown_open = False
        piece_conf_dropdown_open = False
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Grab dropdown
    elif grab_dropdown_rect.collidepoint(mx, my):
        frame_grab_dropdown_open = not frame_grab_dropdown_open
        fps_dropdown_open = False
        camera_dropdown_open = False
        piece_conf_dropdown_open = False
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Camera dropdown
    elif camera_dropdown_rect.collidepoint(mx, my):
        camera_dropdown_open = not camera_dropdown_open
        fps_dropdown_open = False
        frame_grab_dropdown_open = False
        piece_conf_dropdown_open = False
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Piece conf dropdown
    elif piece_conf_dropdown_rect.collidepoint(mx, my):
        piece_conf_dropdown_open = not piece_conf_dropdown_open
        fps_dropdown_open = False
        frame_grab_dropdown_open = False
        camera_dropdown_open = False
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Rotate
    elif rotate_button_rect.collidepoint(mx, my):
        rotate_steps = (rotate_steps + 1) % 4
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Test piece (fire one-shot event)
    if test_piece_button_rect.collidepoint(mx, my):
        try:
            pygame.event.post(pygame.event.Event(pygame.USEREVENT, {"name": "TEST_PIECE"}))
        except Exception:
            pass
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Check camera
    elif camera_button_rect.collidepoint(mx, my):
        if cap is not None:
            cap.release()
            cap = None
        cap = cv2.VideoCapture(selected_camera_index)
        ok, _ = cap.read()
        if ok:
            camera_ready = True
            camera_status = f"Camera {selected_camera_index} OK"
        else:
            camera_ready = False
            camera_status = "Cannot open camera."
            if cap is not None:
                cap.release()
                cap = None
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Segment
    elif segment_button_rect.collidepoint(mx, my):
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            True,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Record
    elif record_button_rect.collidepoint(mx, my):
        if not recording:
            if not camera_ready:
                record_status = "Camera not ready."
                return (
                    cap, camera_ready, camera_status, recording, record_status,
                    fps_dropdown_open, fps_current,
                    frame_grab_dropdown_open, frame_grab_every_current,
                    camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
                    debug_enabled, selected_camera_index, rotate_steps,
                    False,
                    manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
                    manual_offset_x, manual_offset_y
                )
            recording = True
            record_status = "Recording..."
        else:
            recording = False
            record_status = "Stopped."
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Nút Debug — sẽ không bao giờ kích hoạt nếu debug_button_rect=None (đã bỏ nút)
    elif (debug_button_rect is not None) and debug_button_rect.collidepoint(mx, my):
        debug_enabled = not debug_enabled
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # ===== Manual Grid controls =====
    # Nút này giờ chỉ HIỆN/ẨN overlay; manual params luôn áp cho nhận dạng.
    if manual_toggle_rect and manual_toggle_rect.collidepoint(mx, my):
        manual_enabled = not manual_enabled
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    if swap_axes_rect and swap_axes_rect.collidepoint(mx, my):
        manual_swap_axes = not manual_swap_axes
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    if river_minus_rect and river_minus_rect.collidepoint(mx, my):
        manual_river_extra = max(0.0, manual_river_extra - 2.0)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )
    if river_plus_rect and river_plus_rect.collidepoint(mx, my):
        manual_river_extra = min(200.0, manual_river_extra + 2.0)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    if scx_minus_rect and scx_minus_rect.collidepoint(mx, my):
        manual_scale_x = max(0.5, manual_scale_x - 0.02)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )
    if scx_plus_rect and scx_plus_rect.collidepoint(mx, my):
        manual_scale_x = min(1.5, manual_scale_x + 0.02)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    if scy_minus_rect and scy_minus_rect.collidepoint(mx, my):
        manual_scale_y = max(0.5, manual_scale_y - 0.02)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )
    if scy_plus_rect and scy_plus_rect.collidepoint(mx, my):
        manual_scale_y = min(1.5, manual_scale_y + 0.02)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # NEW: Offset X/Y
    if offx_minus_rect and offx_minus_rect.collidepoint(mx, my):
        manual_offset_x = max(-320.0, manual_offset_x - 2.0)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )
    if offx_plus_rect and offx_plus_rect.collidepoint(mx, my):
        manual_offset_x = min(320.0, manual_offset_x + 2.0)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    if offy_minus_rect and offy_minus_rect.collidepoint(mx, my):
        manual_offset_y = max(-320.0, manual_offset_y - 2.0)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )
    if offy_plus_rect and offy_plus_rect.collidepoint(mx, my):
        manual_offset_y = min(320.0, manual_offset_y + 2.0)
        return (
            cap, camera_ready, camera_status, recording, record_status,
            fps_dropdown_open, fps_current,
            frame_grab_dropdown_open, frame_grab_every_current,
            camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
            debug_enabled, selected_camera_index, rotate_steps,
            False,
            manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
            manual_offset_x, manual_offset_y
        )

    # Click ngoài: đóng mọi dropdown
    fps_dropdown_open = False
    frame_grab_dropdown_open = False
    camera_dropdown_open = False
    piece_conf_dropdown_open = False
    return (
        cap, camera_ready, camera_status, recording, record_status,
        fps_dropdown_open, fps_current,
        frame_grab_dropdown_open, frame_grab_every_current,
        camera_dropdown_open, piece_conf_dropdown_open, piece_conf_current,
        debug_enabled, selected_camera_index, rotate_steps,
        False,
        manual_enabled, manual_swap_axes, manual_river_extra, manual_scale_x, manual_scale_y,
        manual_offset_x, manual_offset_y
    )


def draw_record_tab(
    screen,
    test_piece_button_rect,
    panel,
    frame,
    last_board_img,
    pieces,
    has_board,
    clock,
    board_w,
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
    debug_button_rect,
    camera_indices,
    camera_ready,
    debug_enabled,
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
    # ==== Manual grid (mới) ====
    manual_toggle_rect=None,
    swap_axes_rect=None,
    river_minus_rect=None,
    river_plus_rect=None,
    scx_minus_rect=None,
    scx_plus_rect=None,
    scy_minus_rect=None,
    scy_plus_rect=None,
    # NEW offset control rects
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
    # ==== NEW: hiển thị lịch sử nước đi ====
    record_filename="",
    right_panel_rect=None,
):
    """Vẽ UI tab Record trong panel phải + cửa sổ Move Log."""
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
        """Vẽ giá trị nhỏ gọn trong góc phải-dưới của nút bên phải của hàng đó."""
        val = mono.render(text, True, (200, 200, 200))
        vx = right_rect.right - val.get_width() - 8
        vy = right_rect.bottom - val.get_height() - 6
        screen.blit(val, (vx, vy))

    option_h = 28

    # Labels + dropdowns (dùng chính text trong nút/dropdown, không vẽ label riêng)
    _draw_dropdown(screen, mono, fps_dropdown_rect, f"FPS: {fps_current}")
    _draw_dropdown(screen, mono, grab_dropdown_rect, f"Grab every: {frame_grab_every_current}")
    _draw_dropdown(screen, mono, camera_dropdown_rect, f"Camera: {selected_camera_index}")
    _draw_dropdown(screen, mono, piece_conf_dropdown_rect, f"Piece conf: {piece_conf_current:.1f}")

    # Buttons
    _draw_button(screen, mono, camera_button_rect, "Check camera", mouse_pos, active=camera_ready)
    _draw_button(screen, mono, segment_button_rect, "Segment board", mouse_pos, active=board_locked)
    _draw_button(screen, mono, rotate_button_rect, f"Rotate 90° (x{rotate_steps})", mouse_pos, active=(rotate_steps % 4 != 0))
    _draw_button(screen, mono, record_button_rect, "Stop" if recording else "Record", mouse_pos, active=recording)
    # BỎ vẽ nút Debug (không hiển thị nữa)
    if debug_button_rect is not None:
        _draw_button(screen, mono, debug_button_rect, "Debug: ON" if debug_enabled else "Debug: OFF", mouse_pos, active=debug_enabled)
    _draw_button(screen, mono, test_piece_button_rect, "Test piece", mouse_pos, active=False)

    # ==== Manual Grid Section ====
    if manual_toggle_rect is not None:
        # Nút này bây giờ là HIỆN/ẨN overlay
        _draw_button(screen, mono, manual_toggle_rect, f"Grid Overlay: {'SHOW' if manual_enabled else 'HIDE'}", mouse_pos, active=manual_enabled)
    if swap_axes_rect is not None:
        # Luôn cho phép chỉnh (dù overlay ẩn)
        _draw_button(screen, mono, swap_axes_rect, f"Swap Axes: {'Y→9' if manual_swap_axes else 'X→9'}", mouse_pos, active=True)

    # River gap row
    if river_minus_rect and river_plus_rect:
        _draw_button(screen, mono, river_minus_rect, "River -", mouse_pos, active=True)
        _draw_button(screen, mono, river_plus_rect,  "River +", mouse_pos, active=True)
        _draw_value_in_right_btn(river_plus_rect, mono, f"river_extra={manual_river_extra:.0f}px")

    # Scale X/Y rows
    if scx_minus_rect and scx_plus_rect:
        _draw_button(screen, mono, scx_minus_rect, "ScaleX -", mouse_pos, active=True)
        _draw_button(screen, mono, scx_plus_rect,  "ScaleX +", mouse_pos, active=True)
        _draw_value_in_right_btn(scx_plus_rect, mono, f"scale_x={manual_scale_x:.2f}")

    if scy_minus_rect and scy_plus_rect:
        _draw_button(screen, mono, scy_minus_rect, "ScaleY -", mouse_pos, active=True)
        _draw_button(screen, mono, scy_plus_rect,  "ScaleY +", mouse_pos, active=True)
        _draw_value_in_right_btn(scy_plus_rect, mono, f"scale_y={manual_scale_y:.2f}")

    # Offset X/Y rows
    if offx_minus_rect and offx_plus_rect:
        _draw_button(screen, mono, offx_minus_rect, "OffsetX -", mouse_pos, active=True)
        _draw_button(screen, mono, offx_plus_rect,  "OffsetX +", mouse_pos, active=True)
        _draw_value_in_right_btn(offx_plus_rect, mono, f"offset_x={manual_offset_x:.0f}px")

    if offy_minus_rect and offy_plus_rect:
        _draw_button(screen, mono, offy_minus_rect, "OffsetY -", mouse_pos, active=True)
        _draw_button(screen, mono, offy_plus_rect,  "OffsetY +", mouse_pos, active=True)
        _draw_value_in_right_btn(offy_plus_rect, mono, f"offset_y={manual_offset_y:.0f}px")

    # Dropdown options (nếu đang mở)
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

    # ===== NEW: Move Log window ở phần trống dưới cùng =====
    # Tính vùng còn lại dựa trên nút/thanh cuối cùng
    bottoms = []
    for r in [
        offy_plus_rect, offx_plus_rect, scy_plus_rect, scx_plus_rect,
        river_plus_rect, swap_axes_rect, manual_toggle_rect,
        test_piece_button_rect, debug_button_rect, record_button_rect
    ]:
        if r is not None:
            bottoms.append(r.bottom)
    y_start = (max(bottoms) + 14) if bottoms else (panel_origin_y + 14)

    # Nếu có right_panel_rect -> giới hạn khung
    if right_panel_rect is not None:
        x = fps_dropdown_rect.x
        w = fps_dropdown_rect.width
        y = max(y_start, right_panel_rect.y + 10)
        h = max(0, right_panel_rect.bottom - y - 12)
    else:
        # fallback: dùng theo drop-down
        x = fps_dropdown_rect.x
        w = fps_dropdown_rect.width
        y = y_start
        h = 180

    if h > 24:
        # Khung nền tối + viền
        box_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, (28, 28, 28), box_rect, border_radius=6)
        pygame.draw.rect(screen, (90, 90, 90), box_rect, 1, border_radius=6)

        title = mono.render("Move Log", True, (210, 210, 210))
        screen.blit(title, (x + 8, y + 6))

        # Vùng text bên trong (chừa 24px cho title)
        inner_x = x + 8
        inner_y = y + 28
        inner_w = w - 16
        inner_h = h - 36

        # Đọc file log hiện tại rồi hiển thị phần cuối vừa khít
        lines = []
        if record_filename and os.path.isfile(record_filename):
            try:
                with open(record_filename, "r", encoding="utf-8") as f:
                    lines = [ln.rstrip("\n") for ln in f.readlines()]
            except Exception:
                lines = ["(Error reading current play file)"]
        else:
            # Cho biết trạng thái, vẫn hiển thị được khi chưa record
            if record_filename:
                lines = [f"(Waiting for moves in: {os.path.basename(record_filename)})"]
            else:
                lines = ["(Press 'Record' and make a move to start logging)"]

        # Chỉ lấy vừa số dòng hiển thị
        line_h = mono.get_linesize()
        max_lines = max(1, inner_h // line_h)
        if len(lines) > max_lines:
            lines = lines[-max_lines:]

        cy = inner_y
        for s in lines:
            surf = mono.render(s, True, (235, 235, 235))
            screen.blit(surf, (inner_x, cy))
            cy += line_h
