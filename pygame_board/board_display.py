import pygame

# ============================
# Board drawing
# ============================

BG_COLOR = (235, 235, 210)
LINE_COLOR = (60, 60, 60)


def _board_geometry(surface):
    """Tính toán margin + bước lưới theo kích thước surface."""
    w, h = surface.get_size()
    margin_x = int(w * 0.06)
    margin_y = int(h * 0.06)

    grid_w = w - 2 * margin_x
    grid_h = h - 2 * margin_y

    cell_x = grid_w / 8.0  # 9 cột => 8 khoảng
    cell_y = grid_h / 9.0  # 10 hàng => 9 khoảng

    return margin_x, margin_y, cell_x, cell_y


def draw_board(surface: pygame.Surface) -> None:
    """
    Vẽ bàn cờ Tượng Kỳ 9x10 trên surface đã có BG.
    (Không phụ thuộc font, chỉ vẽ line.)
    """
    surface.fill(BG_COLOR)
    w, h = surface.get_size()
    margin_x, margin_y, cell_x, cell_y = _board_geometry(surface)

    left = margin_x
    right = w - margin_x
    top = margin_y
    bottom = h - margin_y

    # Outer rect
    pygame.draw.rect(surface, LINE_COLOR, (left, top, right - left, bottom - top), 2)

    # Horizontal lines (10 hàng)
    for r in range(10):
        y = top + r * cell_y
        pygame.draw.line(surface, LINE_COLOR, (left, y), (right, y), 1)

    # Vertical lines (9 cột) với khoảng sông
    for c in range(9):
        x = left + c * cell_x
        if c == 0 or c == 8:
            # biên ngoài: vẽ full
            pygame.draw.line(surface, LINE_COLOR, (x, top), (x, bottom), 1)
        else:
            # trên
            pygame.draw.line(surface, LINE_COLOR, (x, top), (x, top + 4 * cell_y), 1)
            # dưới
            pygame.draw.line(surface, LINE_COLOR, (x, top + 5 * cell_y), (x, bottom), 1)

    # Hai cung tướng (đường chéo)
    # Trên (đen)
    cx1 = left + 3 * cell_x
    cx2 = left + 5 * cell_x
    y_top = top
    y3 = top + 2 * cell_y
    pygame.draw.line(surface, LINE_COLOR, (cx1, y_top), (cx2, y3), 1)
    pygame.draw.line(surface, LINE_COLOR, (cx2, y_top), (cx1, y3), 1)

    # Dưới (đỏ)
    cx1 = left + 3 * cell_x
    cx2 = left + 5 * cell_x
    y_bot0 = bottom
    y_bot3 = bottom - 2 * cell_y
    pygame.draw.line(surface, LINE_COLOR, (cx1, y_bot0), (cx2, y_bot3), 1)
    pygame.draw.line(surface, LINE_COLOR, (cx2, y_bot0), (cx1, y_bot3), 1)


# ============================
# Piece drawing
# ============================

# Chuẩn hoá loại quân từ label model:
#   "red-rook", "black-knight", "red-guard", "black-general", ...
ROLE_MAP = {
    "rook": "rook",
    "r": "rook",
    "chariot": "rook",

    "horse": "horse",
    "knight": "horse",

    "bishop": "bishop",
    "elephant": "bishop",
    "minister": "bishop",

    "advisor": "advisor",
    "guard": "advisor",

    "king": "king",
    "general": "king",

    "pawn": "pawn",
    "soldier": "pawn",

    "cannon": "cannon",
    "gun": "cannon",
}

# Ký tự Hán cho mỗi (màu, loại)
# Có thể chỉnh lại theo ý nếu muốn.
HAN_CHAR = {
    ("black", "rook"): "車",
    ("black", "horse"): "馬",
    ("black", "bishop"): "象",
    ("black", "advisor"): "士",
    ("black", "king"): "将",
    ("black", "pawn"): "卒",
    ("black", "cannon"): "砲",

    ("red", "rook"): "俥",
    ("red", "horse"): "傌",
    ("red", "bishop"): "相",
    ("red", "advisor"): "仕",
    ("red", "king"): "帥",
    ("red", "pawn"): "兵",
    ("red", "cannon"): "炮",
}


def _parse_label(label: str):
    """
    label: 'red-rook', 'black-knight', 'red-guard-0', ...
    -> (side, role) đã chuẩn hoá.
    """
    side = "red" if label.startswith("red") else "black"
    # lấy phần sau cùng sau dấu '-'
    raw = label.split("-")[-1].lower()
    role = ROLE_MAP.get(raw, raw)
    return side, role


def draw_pieces(surface: pygame.Surface, font: pygame.font.Font, pieces):
    """
    Vẽ quân cờ lên bàn.
    pieces: list[(label, (col, row))], col 0..8, row 0..9.
    """
    if not pieces:
        return

    w, h = surface.get_size()
    margin_x, margin_y, cell_x, cell_y = _board_geometry(surface)

    # Style quân
    radius = int(min(cell_x, cell_y) * 0.38)
    outline = max(1, int(radius * 0.08))

    for label, (c, r) in pieces:
        if not (0 <= c <= 8 and 0 <= r <= 9):
            continue

        side, role = _parse_label(label)
        char = HAN_CHAR.get((side, role), "·")  # KHÔNG BAO GIỜ dùng 'B','R' làm fallback

        cx = margin_x + c * cell_x
        cy = margin_y + r * cell_y

        # Màu theo phe
        if side == "red":
            fill_color = (220, 70, 70)
            text_color = (255, 255, 255)
        else:
            fill_color = (40, 40, 40)
            text_color = (255, 255, 255)

        # Vẽ vòng ngoài trắng giống screenshot
        pygame.draw.circle(surface, (240, 240, 240), (int(cx), int(cy)), radius + outline)
        pygame.draw.circle(surface, fill_color, (int(cx), int(cy)), radius)
        pygame.draw.circle(surface, (240, 240, 240), (int(cx), int(cy)), radius, 2)

        # Render chữ
        text_surf = font.render(char, True, text_color)
        tw, th = text_surf.get_size()
        surface.blit(text_surf, (cx - tw / 2, cy - th / 2))
