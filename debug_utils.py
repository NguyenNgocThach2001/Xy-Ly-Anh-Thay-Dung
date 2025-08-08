import pygame
import cv2

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
        scale = min(1.0, max_w / float(w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        surface = pygame.image.frombuffer(rgb.tobytes(), (new_w, new_h), "RGB")
        return surface.convert()

    @staticmethod
    def draw_pieces_on_board(board_img, pieces):
        if board_img is None:
            return None
        img = board_img.copy()
        h, w = img.shape[:2]
        cell_w = w / 9.0
        cell_h = h / 10.0
        for label, (col, row) in (pieces or []):
            cx = int(col * cell_w + cell_w / 2)
            cy = int(row * cell_h + cell_h / 2)
            cv2.circle(img, (cx, cy), 16, (0, 0, 255), 2)
            cv2.putText(img, str(label), (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return img

    def _title(self, text, y):
        label = self.font.render(text, True, (20, 20, 20))
        self.screen.blit(label, (self.ox, y))
        return y + label.get_height() + 6

    def show(self, raw_bgr, warp_bgr, pieces, pieces_count, has_board, fps_text=""):
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