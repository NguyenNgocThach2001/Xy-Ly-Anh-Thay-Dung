import pygame

# LƯỚI CHUẨN: gốc (0,0) ở GÓC TRÊN-TRÁI của ảnh warp
# - Cột: 0..8 (trái -> phải)
# - Hàng: 0..9 (trên -> dưới)
# - MẶC ĐỊNH: QUÂN ĐEN Ở TRÊN, QUÂN ĐỎ Ở DƯỚI
ROWS, COLS = 10, 9

# ===== Alias role cho các label từ model (soldier/chariot/horse/...) =====
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


def piece_role(name: str):
    """Chuẩn hoá loại quân từ label model (soldier/chariot/horse/... -> role chuẩn)."""
    if not name:
        return None
    raw = name.split("-")[-1].lower()
    return ROLE_MAP.get(raw, raw)


# =========================
# Tiện ích mảng/quân
# =========================
def normalize_pieces(pieces):
    """Kẹp (col,row) trong biên và bỏ trùng, ưu tiên phần tử sau cùng."""
    cell2lab = {}
    for lab, (c, r) in (pieces or []):
        if 0 <= c <= 8 and 0 <= r <= 9:
            cell2lab[(c, r)] = lab
    out = [(lab, (c, r)) for (c, r), lab in cell2lab.items()]
    out.sort(key=lambda x: (x[1][1], x[1][0], x[0]))
    return out


def grid_from_pieces(pieces):
    g = [[None for _ in range(COLS)] for __ in range(ROWS)]
    for lab, (c, r) in (pieces or []):
        if 0 <= c < COLS and 0 <= r < ROWS:
            g[r][c] = lab
    return g


def apply_move(grid, src, dst):
    """Trả về bản sao grid sau khi di từ src->dst (không xét hợp lệ)."""
    sr, sc = src
    dr, dc = dst
    g2 = [row[:] for row in grid]
    g2[dr][dc] = g2[sr][sc]
    g2[sr][sc] = None
    return g2


def diff_grids(prev_grid, new_grid):
    """Danh sách thay đổi (r,c,a,b)."""
    changes = []
    for r in range(ROWS):
        for c in range(COLS):
            a = prev_grid[r][c]
            b = new_grid[r][c]
            if a != b:
                changes.append((r, c, a, b))
    return changes


def is_simple_move(prev_grid, new_grid):
    """
    Một nước đơn, khi:
    - Di thường: đúng 2 cell thay đổi:
        + 1 src: a!=None, b=None
        + 1 dst: a=None,   b!=None
    - Ăn quân: đúng 2 cell thay đổi:
        + 1 src: a!=None, b=None
        + 1 dst: a!=None, b!=None  (ô đích từ quân địch -> quân mình)
    """
    ch = diff_grids(prev_grid, new_grid)
    if len(ch) != 2:
        return (False, None, None, ch)

    src = None
    dst = None
    #a: quan ban tai r c truoc do
    #b: quan ban tai r c moi
    for (r, c, a, b) in ch:
        if a is not None and b is None:
            # ô r,c tại thời điểm a có quân, nhưng thời điểm b thì không có quân
            # => quân a đã di chuyển
            # Vậy ô r-c là điểm nó bắt đầu di chuyển?
            if src is not None: # Chỉ có 1 quân được xuất phát trong 1 nước đi. Nếu source có thì không hợp lệ.
                return (False, None, None, ch)
            src = (r, c)    
        elif a is None and b is not None:
            # Nếu ô r,c tại thời điểm a không có quân mà thời điểm b có quân.
            # => có thằng nó nhảy vào ô này.
            if dst is not None: # Nếu có thằng đã nhảy vào trước đó => không hợp lệ.
                return (False, None, None, ch)
            dst = (r, c)
        elif a is not None and b is not None:
            # Nếu cả 2 thời điểm cùng có quân, tức có 2 con (phải phân thân mới 2 thời điểm 2 thân xác).
            # Nếu lượt trước có thằng đi đến đích, tức nó vừa ăn quân.
            if dst is not None:
                return (False, None, None, ch)
            dst = (r, c)
        else:
            # a=None, b=None => không hợp lệ
            return (False, None, None, ch)
        
    # nếu chỉ có ô ban đầu hoặc có ô đích thôi, vậy nó đi đâu??
    if src is None or dst is None:
        return (False, None, None, ch)
    return (True, src, dst, ch)


def count_nonempty(grid):
    return sum(1 for r in range(ROWS) for c in range(COLS) if grid[r][c] is not None)


# Kiểm tra hợp lệ theo màu
def color_of(name):
    return "red" if name and name.startswith("red-") else ("black" if name and name.startswith("black-") else None)

def same_color(a, b):
    return a and b and color_of(a) == color_of(b)

def path_clear_rook(grid, sr, sc, dr, dc):
    if sr == dr:
        step = 1 if dc > sc else -1
        # step la 1 hoac -1 moi vong lap
        for c in range(sc + step, dc, step):
            if grid[sr][c] is not None:
                return False
        return True
    if sc == dc:
        step = 1 if dr > sr else -1
        for r in range(sr + step, dr, step):
            if grid[r][sc] is not None:
                return False
        return True
    return False


def path_clear_bishop(grid, sr, sc, dr, dc):
    # Tượng đi chéo 2, không qua sông
    if abs(dr - sr) != 2 or abs(dc - sc) != 2:
        return False
    mr = (sr + dr) // 2 # lên xuống 2
    mc = (sc + dc) // 2 # trái phải 2
    return grid[mr][mc] is None


def path_clear_knight(grid, sr, sc, dr, dc):
    vr, vc = dr - sr, dc - sc
    # 8 hướng, chặn chaan
    # diff giua 2 lan di ra duoc huong' di.
    if (vr, vc) == (-2, -1):
        return grid[sr - 1][sc] is None
    if (vr, vc) == (-2, 1):
        return grid[sr - 1][sc] is None
    if (vr, vc) == (2, -1):
        return grid[sr + 1][sc] is None
    if (vr, vc) == (2, 1):
        return grid[sr + 1][sc] is None
    if (vr, vc) == (-1, -2):
        return grid[sr][sc - 1] is None
    if (vr, vc) == (1, -2):
        return grid[sr][sc - 1] is None
    if (vr, vc) == (-1, 2):
        return grid[sr][sc + 1] is None
    if (vr, vc) == (1, 2):
        return grid[sr][sc + 1] is None
    return False


def inside_palace(name, r, c):
    # sĩ, tướng chỉ được đi trong cung.
    if color_of(name) == "red":
        return 7 <= r <= 9 and 3 <= c <= 5
    else:  # black
        return 0 <= r <= 2 and 3 <= c <= 5


def valid_move(grid, name, src, dst):
    """
    Kiểm tra hợp lệ theo **mặc định đen ở TRÊN** (hàng nhỏ -> lớn là đi xuống).
    """
    sr, sc = src
    dr, dc = dst
    if src == dst:
        return False
    target = grid[dr][dc]
    if same_color(name, target):
        return False

    side = color_of(name)
    role = piece_role(name)

    # Xe
    if role == "rook":
        return path_clear_rook(grid, sr, sc, dr, dc)

    # Tượng
    if role == "bishop":
        if not path_clear_bishop(grid, sr, sc, dr, dc):
            return False
        # không qua sông
        if side == "red" and dr < 5:
            return False
        if side == "black" and dr > 4:
            return False
        return True

    # Mã
    if role == "horse":
        vr, vc = dr - sr, dc - sc
        if (abs(vr), abs(vc)) not in ((2, 1), (1, 2)):
            return False
        return path_clear_knight(grid, sr, sc, dr, dc)

    # Sĩ
    if role == "advisor":
        if not inside_palace(name, dr, dc):
            return False
        return abs(dr - sr) == 1 and abs(dc - sc) == 1

    # Tướng
    if role == "king":
        if not inside_palace(name, dr, dc):
            return False
        # đi 1 ô dọc hoặc ngang
        return (abs(dr - sr) == 1 and dc == sc) or (abs(dc - sc) == 1 and dr == sr)

    # Pháo
    if role == "cannon":
        # đường thẳng mới được đi
        if not (sr == dr or sc == dc):
            return False
        cnt = 0
        # mếi đi dọc
        if sr == dr:
            step = 1 if dc > sc else -1
            for c in range(sc + step, dc, step):
                if grid[sr][c] is not None:
                    cnt += 1
        else:
            # nếu đi ngang
            step = 1 if dr > sr else -1
            for r in range(sr + step, dr, step):
                if grid[r][sc] is not None:
                    cnt += 1
        # không qua/ăn sai kiểu
        # nếu không ăn quân và đi không có vật cản thì hợp lệ. Nếu ăn quân mà có 1 vật cản thì hợp lệ.
        return (cnt == 0 and target is None) or (cnt == 1 and target is not None)

    # Tốt
    if role == "pawn":
        if side == "red":
            # đi lên (r giảm)
            if sr >= 5:  # trước sông
                return (dr == sr - 1 and dc == sc)
            else:  # qua sông rồi
                # qua sông thì được đi ngang
                return (dr == sr - 1 and dc == sc) or (dr == sr and abs(dc - sc) == 1)
        else:  # black
            # đi xuống (r tăng)
            if sr <= 4:  # trước sông
                return (dr == sr + 1 and dc == sc)
            else:
                # qua sông thì được đi ngang
                return (dr == sr + 1 and dc == sc) or (dr == sr and abs(dc - sc) == 1)

    return False


# Board engine, validator, vẽ là của board_display
class PygameBoard:
    """
    - Luôn dùng gốc toạ độ chuẩn: đen ở TRÊN, đỏ ở DƯỚI.
    """
    def __init__(self, draw_board_fn, draw_pieces_fn, font, stable_k=3, init_full=True):
        self.draw_board = draw_board_fn # board_display fn
        self.draw_pieces = draw_pieces_fn # board_display fn
        self.font = font
        self.STABLE_K = max(1, int(stable_k)) 

        self.reset()

        if init_full:
            #khoi tao day du
            self.state_live = self._starting_position()
            self.grid_live = grid_from_pieces(self.state_live) # trang thai ban co hien tai
            self.side_to_move = "red"  # đỏ đi trước theo luật
            self._has_new_move_flag = False
            self._last_move_text = None # trang thai nuoc di cuoi cung
            self.dirty = True # danh dau can ve lai

        self._move_log_fh = None # file handle, bien doc ghi file

    # ---- basic state ----
    def reset(self):
        self.state_live = []
        self.grid_live = [[None for _ in range(COLS)] for __ in range(ROWS)] # clear het quan co
        self.side_to_move = "red"
        self.staging_grid = None 
        self.staging_state = None
        self.staging_move = None
        self.staging_count = 0
        self._has_new_move_flag = False
        self._last_move_text = None
        self.dirty = False
    # khoi tao trang thai dau tien cua ban co
    def _starting_position(self):
        pieces = []
        # black top
        pieces += [("black-rook", (0, 0)), ("black-knight", (1, 0)), ("black-bishop", (2, 0)),
                   ("black-guard", (3, 0)), ("black-king", (4, 0)), ("black-guard", (5, 0)),
                   ("black-bishop", (6, 0)), ("black-knight", (7, 0)), ("black-rook", (8, 0))]
        pieces += [("black-cannon", (1, 2)), ("black-cannon", (7, 2))]
        
        for c in range(0, 9, 2):
            pieces.append(("black-pawn", (c, 3)))
        # red bottom
        pieces += [("red-rook", (0, 9)), ("red-knight", (1, 9)), ("red-bishop", (2, 9)),
                   ("red-guard", (3, 9)), ("red-king", (4, 9)), ("red-guard", (5, 9)),
                   ("red-bishop", (6, 9)), ("red-knight", (7, 9)), ("red-rook", (8, 9))]
        pieces += [("red-cannon", (1, 7)), ("red-cannon", (7, 7))]

        for c in range(0, 9, 2):
            pieces.append(("red-pawn", (c, 6)))
        return pieces

    def has_new_move(self):
        return self._has_new_move_flag

    def get_last_move_text(self):
        self._has_new_move_flag = False
        return self._last_move_text

    # internal logging 
    def _log(self, msg):
        print(msg, flush=True)

    # main update 
    def update_state(self, board_locked, detected_pieces):
        """
        Nhận (label,(col,row)) và cập nhật:
          - Nếu chưa có snapshot: gom K khung giống nhau.
          - Nếu đã có: chỉ chấp nhận nước đơn (di 1 quân, có thể ăn) và hợp lệ theo luật.
        """
        #input các quân cờ detect được và vị trí quân cờ, lúc này tên có thể không hợp lệ, chuẩn hóa tên
        detected_pieces = normalize_pieces(detected_pieces)
        #biến thành ma trận [][] cho dễ thao tác
        grid_new = grid_from_pieces(detected_pieces)
        #đếm số quân cờ
        prev_cnt = count_nonempty(self.grid_live)
        new_cnt = count_nonempty(grid_new)

        # 1) Snapshot khởi tạo
        # Lúc đầu khởi tạo bàn cờ thì không có quân nào, trường hợp đặc biệt, sẽ quan sát bàn cờ thực tế và tái tạo lại theo đúng hình hiện có
        if not board_locked or prev_cnt == 0:
            if self.staging_grid is not None:
                if grid_new == self.staging_grid: # bàn cờ trước giống bàn cờ mới.
                    self.staging_count += 1 # count đủ thì cập nhật state mới
                else:
                    self.staging_grid = grid_new
                    self.staging_state = detected_pieces
                    self.staging_move = None
                    self.staging_count = 1
            else:
                self.staging_grid = grid_new
                self.staging_state = detected_pieces
                self.staging_move = None
                self.staging_count = 1

            self._log(f"INIT SNAPSHOT {self.staging_count}/{self.STABLE_K} pieces={new_cnt}")
            #
            if self.staging_count >= self.STABLE_K:
                self.grid_live = self.staging_grid
                self.state_live = self.staging_state[:]
                self.staging_grid = None
                self.staging_state = None
                self.staging_move = None
                self.staging_count = 0
                self.side_to_move = "red"
                self.dirty = True
                self._log("COMMIT INIT SNAPSHOT")
                return True
            return False

        # 2) Nước đi đơn giản (1 quân đi, di chuyển hoặc ăn quân)
        ok_simple, src, dst, changes = is_simple_move(self.grid_live, grid_new)
        # nếu ăn quân hoặc di chuyển, số lượng quân mới và cũ chỉ chênh lệch tối đa 1
        cnt_ok = (new_cnt == prev_cnt) or (new_cnt == prev_cnt - 1)
        if not ok_simple or not cnt_ok:
            # cho nay debug thoi
            self._log(f"REJECT: ok_simple={ok_simple} cnt_ok={cnt_ok} ch_len={len(changes)} prev={prev_cnt} new={new_cnt}")
            if len(changes) <= 8:
                self._log(f"  changes={changes}")
            # return nuoc di khong hop le
            return False

        # src, dst đều [][]
        sr, sc = src
        dr, dc = dst
        mover = self.grid_live[sr][sc]
        # ô di chuyển phải là ô có quân cờ, nếu không có quân thì ai di chuyển? 
        if mover is None:
            return False

        # Sai lượt?
        if color_of(mover) != self.side_to_move:
            self._log(f"REJECT: wrong turn. turn={self.side_to_move}, mover={mover} at {src}->{dst}")
            return False

        # Nước đi đúng quy tắc?
        if not valid_move(self.grid_live, mover, (sr, sc), (dr, dc)):
            self._log(f"INVALID MOVE by {mover}: ({sc}, {sr})->({dc}, {dr})")
            return False

        # 3) Ổn định K khung trước khi chấp nhận
        move_tuple = ((sr, sc), (dr, dc))
        if self.staging_move == move_tuple and self.staging_grid == grid_new:
            self.staging_count += 1
        else:
            self.staging_move = move_tuple
            self.staging_grid = grid_new
            self.staging_state = detected_pieces
            self.staging_count = 1

        self._log(f"CHECK MOVE: mover={mover} src={src} dst={dst} prev={prev_cnt} new={new_cnt} ch_len={len(changes)}")
        if self.staging_count < self.STABLE_K:
            self._log(f"NEW MOVE {src}->{dst} ({self.staging_count}/{self.STABLE_K}) turn={self.side_to_move}")
            return False

        # 4) Sau khi ổn định K khung hình sẽ xác nhận nước đi hợp lệ.
        # Lưu lại quân bị ăn
        captured_before = self.grid_live[dr][dc]

        # Cập nhật board state
        self.grid_live = self.staging_grid
        self.state_live = self.staging_state[:]
        self._has_new_move_flag = True

        # Tạo chuỗi log: quân đi + (quân bị ăn nếu có) + toạ độ
        if captured_before is not None:
            move_desc = f"{mover} x {captured_before}"
        else:
            move_desc = mover
        self._last_move_text = f"{move_desc} ({sc}, {sr})->({dc}, {dr})"

        # đổi lượt
        self.side_to_move = "black" if self.side_to_move == "red" else "red"

        # clear staging
        self.staging_grid = None
        self.staging_state = None
        self.staging_move = None
        self.staging_count = 0

        # Log console chi tiết
        if captured_before is not None:
            self._log(
                f"COMMIT MOVE: {mover} x {captured_before} "
                f"({sc}, {sr})->({dc}, {dr}); next turn={self.side_to_move}"
            )
        else:
            self._log(
                f"COMMIT MOVE: {mover} "
                f"({sc}, {sr})->({dc}, {dr}); next turn={self.side_to_move}"
            )

        return True
