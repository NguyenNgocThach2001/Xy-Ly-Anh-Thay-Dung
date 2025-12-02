import os
import pygame

PLAYS_DIR = "plays"


def load_history_games():
    """
    Đọc tất cả file .txt trong thư mục plays/.
    Mỗi file tương ứng 1 ván (game), trả về list[list[str]].
    """
    games = []
    if not os.path.isdir(PLAYS_DIR):
        return games

    try:
        filenames = sorted(
            [f for f in os.listdir(PLAYS_DIR) if f.lower().endswith(".txt")]
        )
    except Exception:
        return games

    for name in filenames:
        path = os.path.join(PLAYS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            lines = [f"(Error reading {name})\n"]

        if not lines:
            lines = [f"(Empty game file: {name})\n"]

        # prepend tên file để dễ phân biệt (hiện lên dòng đầu)
        if not lines[0].startswith("# "):
            lines = [f"# {name}\n"] + lines
        games.append(lines)

    return games


def draw_history_panel(screen, font, area_rect, games, selected_index):
    """
    Vẽ 1 ván (game) trong danh sách games vào vùng area_rect.
    games: list[list[str]]
    selected_index: index ván hiện tại.
    """
    x, y, w, h = area_rect
    line_height = font.get_linesize()
    max_lines = max(1, h // line_height - 1)

    if not games:
        lines_to_show = ["(Chưa có lịch sử game trong plays/)\n"]
    else:
        idx = max(0, min(selected_index, len(games) - 1))
        lines = games[idx]
        if not lines:
            lines = ["(Ván này trống)\n"]
        lines_to_show = lines[-max_lines:]

    cur_y = y
    for line in lines_to_show:
        text = line.rstrip("\n")
        surf = font.render(text, True, (0, 0, 0))
        screen.blit(surf, (x, cur_y))
        cur_y += line_height
        if cur_y > y + h:
            break


def handle_history_click(mx, my, history_prev_rect, history_next_rect,
                         num_games, selected_index):
    """
    Xử lý click chuột trong tab History: nút < và > để đổi ván.
    Trả về selected_index mới.
    """
    if num_games <= 0:
        return selected_index

    if history_prev_rect.collidepoint(mx, my):
        selected_index = (selected_index - 1) % num_games
    elif history_next_rect.collidepoint(mx, my):
        selected_index = (selected_index + 1) % num_games

    return selected_index


def draw_history_tab(screen,
                     mono,
                     history_prev_rect,
                     history_next_rect,
                     history_area_rect,
                     history_games,
                     history_selected_index):
    """
    Vẽ toàn bộ nội dung tab History (nút prev/next + label + danh sách nước).
    """
    num_games = len(history_games)
    mouse_pos = pygame.mouse.get_pos()

    # Nút prev / next
    hovered_prev = history_prev_rect.collidepoint(mouse_pos)
    hovered_next = history_next_rect.collidepoint(mouse_pos)
    prev_color = (220, 220, 220) if hovered_prev else (200, 200, 200)
    next_color = (220, 220, 220) if hovered_next else (200, 200, 200)

    pygame.draw.rect(screen, prev_color, history_prev_rect)
    pygame.draw.rect(screen, (0, 0, 0), history_prev_rect, 1)
    prev_text = mono.render("<", True, (0, 0, 0))
    prev_rect = prev_text.get_rect(center=history_prev_rect.center)
    screen.blit(prev_text, prev_rect)

    pygame.draw.rect(screen, next_color, history_next_rect)
    pygame.draw.rect(screen, (0, 0, 0), history_next_rect, 1)
    next_text = mono.render(">", True, (0, 0, 0))
    next_rect = next_text.get_rect(center=history_next_rect.center)
    screen.blit(next_text, next_rect)

    # Label hiển thị game hiện tại
    if num_games > 0:
        label = f"Game {history_selected_index + 1} / {num_games}"
    else:
        label = "No games"
    label_surf = mono.render(label, True, (0, 0, 0))
    screen.blit(label_surf, (history_prev_rect.right + 10, history_prev_rect.y + 5))

    # Tiêu đề & nội dung history
    title_surf = mono.render("Game history (folder plays/)", True, (0, 0, 0))
    screen.blit(title_surf, (history_area_rect[0], history_area_rect[1] - 25))
    draw_history_panel(screen, mono, history_area_rect, history_games, history_selected_index)
