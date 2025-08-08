import pygame

ROWS, COLS = 10, 9

def normalize_pieces(pieces):
    cell2lab = {}
    for lab,(c,r) in (pieces or []):
        if 0 <= c <= 8 and 0 <= r <= 9:
            cell2lab[(c,r)] = lab
    out = [(lab,(c,r)) for (c,r),lab in cell2lab.items()]
    out.sort(key=lambda x:(x[1][1], x[1][0], x[0]))
    return out

def grid_from_pieces(pieces):
    g = [[None for _ in range(COLS)] for __ in range(ROWS)]
    for lab,(c,r) in (pieces or []):
        if 0 <= c < COLS and 0 <= r < ROWS:
            g[r][c] = lab
    return g

def diff_grids(A, B):
    ch = []
    for r in range(ROWS):
        for c in range(COLS):
            if A[r][c] != B[r][c]:
                ch.append((r,c,A[r][c],B[r][c]))
    return ch

def is_simple_move(prev_grid, new_grid):
    ch = diff_grids(prev_grid, new_grid)
    if len(ch) == 0: return (False,None,None, ch)
    if len(ch) > 3:  return (False,None,None, ch)
    src = dst = None
    for (r,c,a,b) in ch:
        if a is not None and b is None:
            if src is not None: return (False,None,None, ch)
            src = (r,c)
        elif b is not None:
            if dst is not None: return (False,None,None, ch)
            dst = (r,c)
        else:
            return (False,None,None, ch)
    if src is None or dst is None: return (False,None,None, ch)
    return (True,src,dst, ch)

def count_nonempty(grid):
    return sum(1 for r in range(ROWS) for c in range(COLS) if grid[r][c] is not None)

# ===== Validator cơ bản theo luật hình học =====
def color_of(name): return "red" if name and name.startswith("red-") else ("black" if name else None)
def same_color(a,b): return a and b and color_of(a)==color_of(b)

def path_clear_rook(grid, sr, sc, dr, dc):
    if sr == dr:
        step = 1 if dc>sc else -1
        for c in range(sc+step, dc, step):
            if grid[sr][c] is not None: return False
        return True
    if sc == dc:
        step = 1 if dr>sr else -1
        for r in range(sr+step, dr, step):
            if grid[r][sc] is not None: return False
        return True
    return False

def path_clear_bishop(grid, sr, sc, dr, dc):
    if abs(dr-sr)==2 and abs(dc-sc)==2:
        eye_r=(sr+dr)//2; eye_c=(sc+dc)//2
        return grid[eye_r][eye_c] is None
    return False

def palace_contains(r,c,side):
    return (7<=r<=9 and 3<=c<=5) if side=="red" else (0<=r<=2 and 3<=c<=5)

def valid_move(grid, name, src, dst):
    if name is None: return False
    sr,sc=src; dr,dc=dst
    if sr==dr and sc==dc: return False
    side=color_of(name); target=grid[dr][dc]
    if same_color(name, target): return False
    drd, dcd = dr-sr, dc-sc
    adr, adc = abs(drd), abs(dcd)

    if name.endswith("chariot"):  # xe
        return path_clear_rook(grid, sr, sc, dr, dc)
    if name.endswith("horse"):    # mã
        if (adr,adc) not in [(2,1),(1,2)]: return False
        br = sr + (drd//2) if adr==2 else sr
        bc = sc + (dcd//2) if adc==2 else sc
        return grid[br][bc] is None
    if name.endswith("elephant"): # tượng
        if not path_clear_bishop(grid,sr,sc,dr,dc): return False
        return (side=="red" and dr>=5) or (side=="black" and dr<=4)
    if name.endswith("advisor"):  # sĩ
        return (adr==1 and adc==1) and palace_contains(dr,dc,side)
    if name.endswith("general"):  # tướng
        return (adr+adc)==1 and palace_contains(dr,dc,side)
    if name.endswith("soldier"):  # tốt
        if side=="red":
            if dr>sr: return False
            return (dr==sr-1 and dc==sc) if sr>=5 else ((dr==sr-1 and dc==sc) or (dr==sr and abs(dc-sc)==1))
        else:
            if dr<sr: return False
            return (dr==sr+1 and dc==sc) if sr<=4 else ((dr==sr+1 and dc==sc) or (dr==sr and abs(dc-sc)==1))
    if name.endswith("cannon"):   # pháo
        if not (sr==dr or sc==dc): return False
        cnt=0
        if sr==dr:
            step=1 if dc>sc else -1
            for c in range(sc+step, dc, step):
                if grid[sr][c] is not None: cnt+=1
        else:
            step=1 if dr>sr else -1
            for r in range(sr+step, dr, step):
                if grid[r][sc] is not None: cnt+=1
        return (cnt==0 and target is None) or (cnt==1 and target is not None)
    return False

class PygameBoard:
    """
    Pygame là nơi quyết định update:
    - Nhận (has_board, pieces) mỗi frame.
    - INIT: commit ngay nếu đủ quân (>=32) khi state_live đang rỗng.
    - Sau đó: chỉ nhận 'nước đi đơn giản' + (đếm quân == hoặc == -1).
    - Có debounce K frame; có tolerance & force commit để tránh kẹt.
    """
    def __init__(self, draw_board_fn, draw_pieces_fn, font, stable_k=3, init_full=True, use_validator=True):
        self.draw_board = draw_board_fn
        self.draw_pieces = draw_pieces_fn
        self.font = font
        self.STABLE_K = stable_k
        self.use_validator = use_validator

        self.state_live = self._starting_position() if init_full else []
        self.grid_live  = grid_from_pieces(self.state_live)

        self.staging_grid = None
        self.staging_state = None
        self.staging_move = None
        self.staging_count = 0

        # chống kẹt: theo dõi lặp lại 1 trạng thái grid
        self.last_hash = None
        self.same_hash_count = 0
        self.FORCE_AFTER = 15  # nếu thấy cùng grid 15 lần mà chưa commit -> force

        self.dirty = True  # vẽ ngay khởi tạo

    # Thế cờ chuẩn (nếu init_full=True)
    def _starting_position(self):
        s=[]
        s += [("black-chariot",(0,0)),("black-horse",(1,0)),("black-elephant",(2,0)),
              ("black-advisor",(3,0)),("black-general",(4,0)),("black-advisor",(5,0)),
              ("black-elephant",(6,0)),("black-horse",(7,0)),("black-chariot",(8,0))]
        s += [("black-cannon",(1,2)),("black-cannon",(7,2))]
        s += [("black-soldier",(0,3)),("black-soldier",(2,3)),("black-soldier",(4,3)),
              ("black-soldier",(6,3)),("black-soldier",(8,3))]
        s += [("red-chariot",(0,9)),("red-horse",(1,9)),("red-elephant",(2,9)),
              ("red-advisor",(3,9)),("red-general",(4,9)),("red-advisor",(5,9)),
              ("red-elephant",(6,9)),("red-horse",(7,9)),("red-chariot",(8,9))]
        s += [("red-cannon",(1,7)),("red-cannon",(7,7))]
        s += [("red-soldier",(0,6)),("red-soldier",(2,6)),("red-soldier",(4,6)),
              ("red-soldier",(6,6)),("red-soldier",(8,6))]
        return s

    def _log(self, s): print(s, flush=True)

    def render(self, screen):
        if not self.dirty:
            return False
        screen.fill((245,245,220))
        self.draw_board(screen)
        self.draw_pieces(screen, self.font, self.state_live)
        pygame.display.flip()
        self.dirty = False
        return True

    def update_state(self, has_board: bool, detected_pieces):
        if not has_board:
            self._log("NO_BOARD")
            return False

        detected_pieces = normalize_pieces(detected_pieces or [])
        n = len(detected_pieces)
        self._log(f"BOARD_OK -> {n} pieces")
        if n == 0:
            return False

        grid_new = grid_from_pieces(detected_pieces)

        # hash để theo dõi lặp
        h = tuple(tuple(row) for row in grid_new)
        if h == self.last_hash:
            self.same_hash_count += 1
        else:
            self.last_hash = h
            self.same_hash_count = 1

        # INIT: nếu live rỗng -> commit ngay khi đủ quân
        if len(self.state_live) == 0 and n >= 32:
            self.state_live = detected_pieces[:]
            self.grid_live  = grid_new
            self.dirty = True
            self._log(f"INIT BOARD with {n} pieces")
            return True

        # Không thay đổi
        if grid_new == self.grid_live:
            self._log("NOT NEW STATE")
            return False

        # kiểm tra move đơn giản + count
        ok_simple, src, dst, changes = is_simple_move(self.grid_live, grid_new)
        prev_cnt = count_nonempty(self.grid_live)
        new_cnt  = count_nonempty(grid_new)
        cnt_ok   = (new_cnt == prev_cnt) or (new_cnt == prev_cnt - 1)

        ch_len = len(changes)
        if not ok_simple or not cnt_ok:
            self._log(f"REJECT: ok_simple={ok_simple} cnt_ok={cnt_ok} ch_len={ch_len} prev={prev_cnt} new={new_cnt}")
            if ch_len <= 8:
                self._log(f"  changes={changes[:8]}")
            # Tolerance khi cùng một move đang mồi
            # (sẽ xử lý phía dưới; nếu không cùng move thì dừng luôn)
            if not ok_simple:
                return False

        # Nếu không đạt cnt_ok nhưng là cùng move đang mồi -> cho nới 1 chút
        TOL_COUNT = 2           # cho phép thiếu tối đa 2 quân khi đang mồi cùng 1 move
        MAX_CHANGES_OK = 6      # chấp nhận tối đa 6 thay đổi khi đã "nhìn" cùng move
        tolerated = False

        if not cnt_ok:
            if self.staging_move == (src, dst) and (prev_cnt - new_cnt) <= TOL_COUNT and ch_len <= MAX_CHANGES_OK:
                tolerated = True
                self._log(f"TOLERATE COUNT: prev={prev_cnt} new={new_cnt} ch_len={ch_len}")
            else:
                return False

        # Validate theo loại quân đang ở src trong state_live
        sr, sc = src; dr, dc = dst
        mover = self.grid_live[sr][sc]
        self._log(f"CHECK MOVE: mover={mover} src={src} dst={dst} prev={prev_cnt} new={new_cnt} ch_len={ch_len}")
        if self.use_validator and not valid_move(self.grid_live, mover, src, dst):
            self._log(f"INVALID MOVE by {mover}: {src}->{dst}")
            self.staging_grid = None; self.staging_state=None; self.staging_move=None; self.staging_count=0
            return False

        # Debounce cùng một move
        mv = (src, dst)
        if self.staging_grid is not None and self.staging_move == mv:
            self.staging_count += 1
        else:
            self.staging_grid = grid_new
            self.staging_state = detected_pieces
            self.staging_move = mv
            self.staging_count = 1

        self._log(f"NEW MOVE {src}->{dst} ({self.staging_count}/{self.STABLE_K}) tolerated={tolerated}")

        # Force commit nếu lặp lại quá nhiều lần (tránh kẹt)
        if self.same_hash_count >= self.FORCE_AFTER:
            self.grid_live  = grid_new
            self.state_live = detected_pieces[:]
            self.staging_grid = None; self.staging_state=None; self.staging_move=None; self.staging_count=0
            self.dirty = True
            self._log("FORCE COMMIT after repeat")
            return True

        if self.staging_count >= self.STABLE_K:
            self.grid_live  = self.staging_grid
            self.state_live = self.staging_state[:]
            self.staging_grid = None; self.staging_state=None; self.staging_move=None; self.staging_count=0
            self.dirty = True
            self._log(f"COMMIT MOVE: {src}->{dst}")
            return True

        return False
