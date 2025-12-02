# New replay_panel.py implementing replay UI and logic (simplified)
import os
import pygame
from typing import List, Tuple

# ---- Module state (kept inside this file) ----
_state = {
    "files": [],                 # list of (path, name)
    "selected_name": "",         # basename đang chọn trong list
    "moves": [],                 # parsed moves [((sr,sc),(dr,dc)), ...]
    "step": 0,                   # current step index (0..len)
    "auto": False,               # autoplay flag
    "speed_ms": 800,             # autoplay interval (cố định; không còn nút +/-)
    "last_tick": 0,              # last tick timestamp (ms)
    # cached rects for click handling (computed in draw)
    "rects": {},
}

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


# ------------- File discovery -------------
def _scan_files():
    os.makedirs(PLAYS_DIR, exist_ok=True)
    files = []
    for name in sorted(os.listdir(PLAYS_DIR)):
        if name.lower().endswith(".txt"):
            files.append((os.path.join(PLAYS_DIR, name), name))
    _state["files"] = files
    if files and not _state["selected_name"]:
        _state["selected_name"] = files[-1][1]
        _load_moves_by_name(_state["selected_name"])


# ------------- Internal load -------------
def _load_moves_by_name(name):
    if not name:
        _state["moves"] = []
        _state["step"] = 0
        return
    path = os.path.join(PLAYS_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        moves = parse_replay_moves(lines)
    except Exception:
        moves = []
    _state["moves"] = moves
    _state["step"] = 0


# ------------- Public API used by game_app -------------
def handle_replay_click(mx, my, rect_playback, replay_moves, replay_index, replay_auto):
    """
    Xử lý click trong tab Replay.
    rect_playback: dict có các rect 'prev', 'next', 'auto'.
    Trả về bộ (replay_moves, replay_index, replay_auto).
    """
    _scan_files()
    rects = _state.get("rects", {})

    # Prev / Next
    if rect_playback.get("prev") and rect_playback["prev"].collidepoint(mx, my):
        replay_index = max(0, replay_index - 1)
        _state["step"] = replay_index
    elif rect_playback.get("next") and rect_playback["next"].collidepoint(mx, my):
        replay_index = min(len(replay_moves), replay_index + 1)
        _state["step"] = replay_index
    # Auto toggle
    elif rect_playback.get("auto") and rect_playback["auto"].collidepoint(mx, my):
        replay_auto = not replay_auto
        _state["auto"] = replay_auto
        _state["last_tick"] = pygame.time.get_ticks()
    # Click vào item trong file list -> auto load
    else:
        for r, name in rects.get("file_items", []):
            if r.collidepoint(mx, my):
                _state["selected_name"] = name
                _load_moves_by_name(name)
                replay_moves = _state["moves"][:]
                replay_index = 0
                _state["step"] = 0
                break

    return replay_moves, replay_index, replay_auto


def get_replay_state():
    """
    Trả về (moves, step, auto) để game_app render trên board chính.
    """
    moves = _state.get("moves", [])[:]
    step = max(0, min(_state.get("step", 0), len(moves)))
    auto = bool(_state.get("auto", False))
    return moves, step, auto


def draw_replay_tab(screen, mono, replay_moves, replay_index, replay_auto, record_filename, rect_playback):
    """
    Vẽ tab Replay TỐI GIẢN:
      - Hàng 1: Prev / Next / Auto
      - Hàng 2: Step {idx}/{total} (hiển thị trên Board bên trái)
      - Phần còn lại: Danh sách file (click để load ngay)
    Không còn ô nhập & nút Replay; không còn +/- tốc độ.
    """
    _scan_files()

    # đồng bộ auto state với biến bên ngoài
    _state["auto"] = bool(replay_auto)

    # Anchors
    prev_rect = rect_playback.get("prev")
    next_rect = rect_playback.get("next")
    auto_rect = rect_playback.get("auto")
    if not (prev_rect and next_rect and auto_rect):
        return

    x = prev_rect.x
    w = prev_rect.width + 10 + next_rect.width

    # Row 1: Prev / Next / Auto
    for r, label in [(prev_rect, "Prev"), (next_rect, "Next")]:
        pygame.draw.rect(screen, (70, 70, 70), r, border_radius=6)
        pygame.draw.rect(screen, (120, 120, 120), r, 1, border_radius=6)
        screen.blit(mono.render(label, True, (255, 255, 255)), (r.x + 10, r.y + 6))

    pygame.draw.rect(screen, (70, 70, 70), auto_rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), auto_rect, 1, border_radius=6)
    auto_label = "Auto: ON" if replay_auto else "Auto: OFF"
    screen.blit(mono.render(auto_label, True, (255, 255, 255)), (auto_rect.x + 10, auto_rect.y + 6))

    # Row 2: Step info
    moves = _state["moves"][:] if _state["moves"] else replay_moves
    total = len(moves)
    idx = max(0, min(_state.get("step", replay_index), total))
    info_rect = pygame.Rect(x, auto_rect.bottom + 12, w, auto_rect.height)
    pygame.draw.rect(screen, (245, 245, 245), info_rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), info_rect, 1, border_radius=6)
    info = mono.render(f"Step {idx}/{total}  (hiển thị trên Board bên trái)", True, (0, 0, 0))
    screen.blit(info, (info_rect.x + 8, info_rect.y + 6))

    # File list occupies the rest
    list_y = info_rect.bottom + 12
    avail_h = max(0, (prev_rect.bottom + 4000) - list_y)
    list_rect = pygame.Rect(x, list_y, w, avail_h)
    pygame.draw.rect(screen, (245, 245, 245), list_rect, border_radius=6)
    pygame.draw.rect(screen, (120, 120, 120), list_rect, 1, border_radius=6)

    files = _state["files"]
    item_h = auto_rect.height
    ycur = list_rect.y + 6
    file_item_rects = []
    for _full, name in files:
        r = pygame.Rect(list_rect.x + 6, ycur, list_rect.w - 12, item_h - 2)
        base = (230, 230, 230) if name == _state.get("selected_name") else (250, 250, 250)
        pygame.draw.rect(screen, base, r, border_radius=6)
        pygame.draw.rect(screen, (180, 180, 180), r, 1, border_radius=6)
        screen.blit(mono.render(name, True, (30, 30, 30)), (r.x + 8, r.y + 6))
        file_item_rects.append((r, name))
        ycur += item_h

    # Lưu rects phục vụ click
    _state["rects"] = {
        "file_items": file_item_rects,
    }

    # ---- Autoplay timing ----
    if _state["auto"] and total > 0:
        now = pygame.time.get_ticks()
        if now - _state.get("last_tick", 0) >= max(50, _state["speed_ms"]):
            _state["last_tick"] = now
            idx = min(total, idx + 1)
            _state["step"] = idx
    else:
        _state["step"] = idx
