import pygame
import sys

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 880
BOARD_ROWS = 10
BOARD_COLS = 9
SQUARE_SIZE = 80
MARGIN_TOP = 40

WHITE = (245, 245, 220)
BLACK = (30, 30, 30)
LINE_COLOR = (50, 50, 50)
FONT_SIZE = 28

PIECE_SYMBOLS = {
    "red-general": "帥", "red-advisor": "仕", "red-elephant": "相", "red-horse": "傌",
    "red-chariot": "俥", "red-cannon": "炮", "red-soldier": "兵",
    "black-general": "將", "black-advisor": "士", "black-elephant": "象", "black-horse": "馬",
    "black-chariot": "車", "black-cannon": "砲", "black-soldier": "卒",
}

def draw_board(screen):
    screen.fill(WHITE)
    for row in range(BOARD_ROWS):
        y = MARGIN_TOP + row * SQUARE_SIZE
        pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, y), (SQUARE_SIZE * BOARD_COLS, y), 2)
    for col in range(BOARD_COLS):
        x = SQUARE_SIZE + col * SQUARE_SIZE
        pygame.draw.line(screen, LINE_COLOR, (x, MARGIN_TOP), (x, MARGIN_TOP + 4 * SQUARE_SIZE), 2)
        pygame.draw.line(screen, LINE_COLOR, (x, MARGIN_TOP + 5 * SQUARE_SIZE), (x, MARGIN_TOP + 9 * SQUARE_SIZE), 2)
    for (start, end) in [((3, 0), (5, 2)), ((5, 0), (3, 2)), ((3, 7), (5, 9)), ((5, 7), (3, 9))]:
        x1 = SQUARE_SIZE + start[0] * SQUARE_SIZE
        y1 = MARGIN_TOP + start[1] * SQUARE_SIZE
        x2 = SQUARE_SIZE + end[0] * SQUARE_SIZE
        y2 = MARGIN_TOP + end[1] * SQUARE_SIZE
        pygame.draw.line(screen, LINE_COLOR, (x1, y1), (x2, y2), 2)

def draw_pieces(screen, font, pieces):
    for name, (col, row) in pieces:
        x = SQUARE_SIZE + col * SQUARE_SIZE
        y = MARGIN_TOP + row * SQUARE_SIZE
        symbol = PIECE_SYMBOLS.get(name, "?")
        color = (255, 0, 0) if "red" in name else (0, 0, 0)
        pygame.draw.circle(screen, color, (x, y), 30, 2)
        text = font.render(symbol, True, color)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
