# New replay_panel.py implementing replay UI and logic (simplified)
import os
import pygame
from typing import List, Tuple
from app_state import AppState

PLAYS_DIR = "plays"


# ------------- Move parsing -------------
def parse_replay_moves(lines: List[str]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Parse danh sách dòng log thành list các nước đi:
      [((sr, sc), (dr, dc)), ...]  với sr/dr = row, sc/dc = col.

    LƯU Ý: File log của hệ thống ghi toạ độ ở dạng (col, row), ví dụ:
           "... (1, 7)->(1, 0)"  (col=1,row=7)  -> cần đổi thành ((7,1)->(0,1))
           Tham chiếu: game_app/PygameBoard ghi "({sc}, {sr})->({dc}, {dr})".

    Hỗ trợ các định dạng phổ biến:
      - "sr,sc -> dr,dc"
      - "(sr,sc)->(dr,dc)"
      - "sr sc dr dc"
      - "sr sc -> dr dc"
    Bỏ qua dòng trống hoặc dòng không hợp lệ.
    """
    import re
    moves = []
    for line in lines or []:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        nums = re.findall(r"-?\d+", s)
        if len(nums) >= 4:
            try:
                # File đang lưu (col, row) -> (c1, r1, c2, r2)
                c1, r1, c2, r2 = map(int, nums[:4])
                # Kiểm tra biên hợp lệ theo (col,row)
                if 0 <= c1 <= 8 and 0 <= r1 <= 9 and 0 <= c2 <= 8 and 0 <= r2 <= 9:
                    # Trả về theo hệ (row, col)
                    moves.append(((r1, c1), (r2, c2)))
            except Exception:
                continue
    return moves


# find replay logs
def scan_files(state: AppState):
    import os
    files = []
    os.makedirs("plays", exist_ok=True)
    for name in sorted(os.listdir("plays")):
        if name.lower().endswith(".txt"):
            files.append(name)
    state.replay_files = files
    if files and not state.replay_selected_name:
        state.replay_selected_name = files[-1]
        load_moves_by_name(state, state.replay_selected_name)


# move load
def load_moves_by_name(state: AppState, name: str):
    if not name:
        state.replay_moves = []
        state.replay_step = 0
        return
    path = os.path.join("plays", name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        moves = parse_replay_moves(lines)
    except Exception:
        moves = []
    state.replay_moves = moves
    state.replay_step = 0

def handle_replay_click(mx, my, rect_playback, state: AppState):
    scan_files(state)
    rects = state.replay_rects

    # Prev / Next
    if rect_playback.get("prev") and rect_playback["prev"].collidepoint(mx, my):
        state.replay_step = max(0, state.replay_step - 1)
    elif rect_playback.get("next") and rect_playback["next"].collidepoint(mx, my):
        state.replay_step = min(len(state.replay_moves), state.replay_step + 1)
    elif rect_playback.get("auto") and rect_playback["auto"].collidepoint(mx, my):
        state.replay_auto = not state.replay_auto
        state.replay_last_tick = pygame.time.get_ticks()
    else:
        for r, name in rects.get("file_items", []):
            if r.collidepoint(mx, my):
                state.replay_selected_name = name
                load_moves_by_name(state, name)
                state.replay_step = 0
                break

def get_replay_state(state: AppState):
    moves = state.replay_moves[:]
    step = max(0, min(state.replay_step, len(moves)))
    auto = bool(state.replay_auto)
    return moves, step, auto
# ve replay
def draw_replay_tab(screen, mono, state: AppState, rect_playback):
    scan_files(state)
    state.replay_auto = bool(state.replay_auto)

    prev_rect = rect_playback.get("prev")
    next_rect = rect_playback.get("next")
    auto_rect = rect_playback.get("auto")
    if not (prev_rect and next_rect and auto_rect):
        return

    x = prev_rect.x
    w = prev_rect.width + 10 + next_rect.width

    for r, label in [(prev_rect, "Prev"), (next_rect, "Next")]:
        pygame.draw.rect(screen, (70, 70, 70), r, border_radius=6)
        pygame.draw.rect(screen, (120, 120, 120), r, 1, border_radius=6)
        screen.blit(mono.render(label, True, (255, 255, 255)), (r.x + 10, r.y + 6))

    pygame.draw.rect(screen, (70, 70, 70), auto_rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), auto_rect, 1, border_radius=6)
    auto_label = "Auto: ON" if state.replay_auto else "Auto: OFF"
    screen.blit(mono.render(auto_label, True, (255, 255, 255)), (auto_rect.x + 10, auto_rect.y + 6))

    moves = state.replay_moves
    total = len(moves)
    idx = max(0, min(state.replay_step, total))
    info_rect = pygame.Rect(x, auto_rect.bottom + 12, w, auto_rect.height)
    pygame.draw.rect(screen, (245, 245, 245), info_rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), info_rect, 1, border_radius=6)
    info = mono.render(f"Step {idx}/{total}  (hiển thị trên Board bên trái)", True, (0, 0, 0))
    screen.blit(info, (info_rect.x + 8, info_rect.y + 6))

    list_y = info_rect.bottom + 12
    avail_h = max(0, (prev_rect.bottom + 4000) - list_y)
    list_rect = pygame.Rect(x, list_y, w, avail_h)
    pygame.draw.rect(screen, (245, 245, 245), list_rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), list_rect, 1, border_radius=6)

    files = state.replay_files
    item_h = auto_rect.height
    ycur = list_rect.y + 6
    file_item_rects = []
    for name in files:
        r = pygame.Rect(list_rect.x + 6, ycur, list_rect.w - 12, item_h - 2)
        base = (230, 230, 230) if name == state.replay_selected_name else (250, 250, 250)
        pygame.draw.rect(screen, base, r, border_radius=6)
        pygame.draw.rect(screen, (180, 180, 180), r, 1, border_radius=6)
        screen.blit(mono.render(name, True, (30, 30, 30)), (r.x + 8, r.y + 6))
        file_item_rects.append((r, name))
        ycur += item_h

    # Lưu file_item_rects vào state
    state.replay_rects["file_items"] = file_item_rects

    # ---- Autoplay timing ----
    if state.replay_auto and total > 0:
        now = pygame.time.get_ticks()
        if now - state.replay_last_tick >= max(50, state.replay_speed_ms):
            state.replay_last_tick = now
            idx = min(total, idx + 1)
            state.replay_step = idx
    else:
        state.replay_step = idx