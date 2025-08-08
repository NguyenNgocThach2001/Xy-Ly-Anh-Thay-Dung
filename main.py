import cv2
import pygame
import sys

from detection.board_detection import load_board_model, detect_board_with_mode
from detection.piece_detection import load_piece_model, detect_pieces_and_get_positions
from pygame_board.board_display import draw_board, draw_pieces
from debug_utils import DebugPanel
from pygame_board.pygame_board import PygameBoard

# ===== Config =====
FPS = 15
PIECE_CONF = 0.5
PIECE_IOU  = 0.4
IMG_SIZE   = 640
ACQUIRE_EVERY = 15   # detect_board mỗi 5 frame trong RUN

# ===== Pygame =====
pygame.init()
BOARD_W, BOARD_H = 800, 880
PANEL_W = 300
SCREEN = pygame.display.set_mode((BOARD_W + PANEL_W, BOARD_H))
pygame.display.set_caption("Cờ Tướng - Demo (5 FPS, Live Debug)")
font = pygame.font.SysFont("simsun", 28)
mono = pygame.font.SysFont("consolas", 16)
clock = pygame.time.Clock()

panel = DebugPanel(SCREEN, origin=(BOARD_W + 10, 10), panel_w=PANEL_W - 20, font=mono)
ui = PygameBoard(draw_board, draw_pieces, font, stable_k=3, init_full=True, use_validator=True)
ui.render(SCREEN)  # vẽ khởi tạo (thế cờ chuẩn)

# ===== Models =====
board_model = load_board_model()
piece_model = load_piece_model()

# ===== Camera =====
cap = cv2.VideoCapture("http://192.168.1.105:4747/video")  # đổi IP theo máy bạn
if not cap.isOpened():
    print("❌ Không mở được webcam."); sys.exit()

mode = "BOOTSTRAP"
ok_board = miss_board = 0
last_board_img = None
last_grid_info = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release(); pygame.quit(); sys.exit()

    # --- acquire board ---
    has_board = False
    if mode in ("BOOTSTRAP", "REACQUIRE"):
        board_img, has_board, grid_info = detect_board_with_mode(frame, mode="yolo", model=board_model)
        if has_board:
            ok_board += 1; miss_board = 0
            last_board_img = board_img; last_grid_info = grid_info
            if ok_board >= 2:
                mode = "RUN"; ok_board = 0; miss_board = 0
        else:
            miss_board += 1; ok_board = 0
    else:  # RUN
        if frame_idx % ACQUIRE_EVERY == 0:
            board_img, has_board, grid_info = detect_board_with_mode(frame, mode="yolo", model=board_model)
            if has_board:
                last_board_img = board_img; last_grid_info = grid_info; miss_board = 0
            else:
                miss_board += 1
                if miss_board >= 10:
                    mode = "REACQUIRE"; ok_board = 0

    # --- detect pieces if board available ---
    pieces = []
    if last_board_img is not None and last_grid_info is not None:
        has_board = True
        pieces = detect_pieces_and_get_positions(
            piece_model, last_board_img, last_grid_info,
            conf=PIECE_CONF, iou=PIECE_IOU, imgsz=IMG_SIZE
        )

    # --- let Pygame decide & render left side if changed ---
    committed = ui.update_state(has_board, pieces)
    if committed:
        ui.render(SCREEN)

    # --- always draw debug panel (right) ---
    # clear right panel area
    SCREEN.fill((245,245,220), rect=pygame.Rect(BOARD_W, 0, PANEL_W, BOARD_H))
    panel.show(frame, last_board_img, pieces, len(pieces), has_board, fps_text=f"{clock.get_fps():.1f} FPS")
    pygame.display.update(pygame.Rect(BOARD_W, 0, PANEL_W, BOARD_H))


    clock.tick(FPS)

cap.release()
pygame.quit()
