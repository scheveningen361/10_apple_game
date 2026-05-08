"""
macro2.py
---------
Usage:
  python macro2.py <target_score>

Flow:
  Loop:
    1. Click Play button
    2. Capture screen & recognize board
    3. Solve with Go AnA solver
    4. If solved score >= target: execute moves, exit
    5. Else: click Reset button, repeat
"""

import os, sys, time, json, subprocess
import mss
import cv2
import numpy as np
import pyautogui
from PIL import Image

# ── Settings ──────────────────────────────────────────────────────────────────
PLAY_BUTTON_IMAGE        = "play_button.png"
RESET_BUTTON_IMAGE       = "reset_button.png"
CONFIG_FILE              = "config.json"
TEMPLATES_DIR            = "templates"
GAME_BOARD_TEMPLATE_FILE = "game_board_template.png"
GRID_ROWS, COLS          = 10, 17
GO_SOLVER_EXE            = os.path.join(os.path.dirname(__file__), "apple_solver.exe")

# ── Utilities ─────────────────────────────────────────────────────────────────

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

def load_templates():
    templates = {}
    for i in range(1, 10):
        path = os.path.join(TEMPLATES_DIR, f"{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates[i] = img
    if not templates:
        print("Error: No templates found in", TEMPLATES_DIR)
        sys.exit(1)
    return templates

def capture_screen():
    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[0])
        pil = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)

def find_board(screenshot_cv):
    if not os.path.exists(GAME_BOARD_TEMPLATE_FILE):
        print("Error: game_board_template.png not found.")
        sys.exit(1)
    tmpl = cv2.imread(GAME_BOARD_TEMPLATE_FILE, cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(screenshot_cv, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < 0.8:
        print(f"Error: board not found (match={score:.2f})")
        sys.exit(1)
    return {"left": loc[0], "top": loc[1], "width": tmpl.shape[1], "height": tmpl.shape[0]}

def recognize_board(screenshot_cv, area, templates):
    crop = screenshot_cv[
        area["top"]:area["top"]+area["height"],
        area["left"]:area["left"]+area["width"]
    ]
    board = [[0]*COLS for _ in range(GRID_ROWS)]
    cw = crop.shape[1] / COLS
    ch = crop.shape[0] / GRID_ROWS
    for r in range(GRID_ROWS):
        for c in range(COLS):
            cell = crop[int(r*ch):int((r+1)*ch), int(c*cw):int((c+1)*cw)]
            best, num = -1, 0
            for n, t in templates.items():
                if cell.shape[0] < t.shape[0] or cell.shape[1] < t.shape[1]:
                    continue
                _, v, _, _ = cv2.minMaxLoc(cv2.matchTemplate(cell, t, cv2.TM_CCOEFF_NORMED))
                if v > best:
                    best, num = v, n
            board[r][c] = num
    return board

def click_play():
    try:
        loc = pyautogui.locateOnScreen(PLAY_BUTTON_IMAGE, confidence=0.9)
        if loc:
            pyautogui.click(loc)
            time.sleep(1.5)
            return True
    except Exception as e:
        print(f"Play button error: {e}")
    print("Play button not found.")
    return False

def click_reset():
    try:
        loc = pyautogui.locateOnScreen(RESET_BUTTON_IMAGE, confidence=0.9)
        if loc:
            pyautogui.click(loc)
            time.sleep(1.0)
            return True
    except Exception as e:
        print(f"Reset button error: {e}")
    print("Reset button not found.")
    return False

def solve_with_go(board):
    flat = " ".join(str(board[r][c]) for r in range(GRID_ROWS) for c in range(COLS))
    result = subprocess.run(
        [GO_SOLVER_EXE, "-solver"],
        input=flat,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print("Go solver error:", result.stderr)
        sys.exit(1)
    lines = result.stdout.strip().splitlines()
    moves = []
    score = 0
    for line in lines:
        if line.startswith("MOVES"):
            pass
        elif line.startswith("SCORE"):
            score = int(line.split()[1])
        else:
            parts = line.split()
            if len(parts) == 4:
                moves.append(tuple(map(int, parts)))
    return moves, score

def tighten_move(move, board):
    r1, c1, r2, c2 = move
    min_r, min_c = r2, c2
    max_r, max_c = r1, c1
    found = False
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if board[r][c] != 0:
                if r < min_r: min_r = r
                if c < min_c: min_c = c
                if r > max_r: max_r = r
                if c > max_c: max_c = c
                found = True
    return (min_r, min_c, max_r, max_c) if found else move

def apply_move(move, board):
    r1, c1, r2, c2 = move
    board = [row[:] for row in board]
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            board[r][c] = 0
    return board

def perform_drag(move, board_area, cell_dims, offset):
    r1, c1, r2, c2 = move
    sx = board_area["left"] + c1 * cell_dims[0] + offset[0]
    sy = board_area["top"]  + r1 * cell_dims[1] + offset[1]
    ex = board_area["left"] + (c2+1) * cell_dims[0] + offset[0]
    ey = board_area["top"]  + (r2+1) * cell_dims[1] + offset[1]
    pyautogui.moveTo(sx, sy, duration=0.08)
    pyautogui.mouseDown()
    pyautogui.moveTo(ex, ey, duration=0.08)
    pyautogui.moveRel(8,  0, duration=0.02)
    pyautogui.moveRel(-16, 0, duration=0.02)
    pyautogui.moveRel(8,  0, duration=0.02)
    time.sleep(0.08)
    pyautogui.mouseUp()
    time.sleep(0.25)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python macro2.py <target_score>")
        sys.exit(1)

    target = int(sys.argv[1])
    print(f"=== macro2 : repeat until score >= {target} ===")

    if not os.path.exists(GO_SOLVER_EXE):
        print(f"Error: {GO_SOLVER_EXE} not found.")
        sys.exit(1)

    cfg = load_config()
    offset = (cfg.get("mouse_offset_x", 0), cfg.get("mouse_offset_y", 0))
    templates = load_templates()

    print("Starting in 3 seconds...")
    time.sleep(3)

    attempt = 0
    while True:
        attempt += 1

        # 1. Click Play
        if not click_play():
            sys.exit(1)

        # 2. Capture & recognize board
        time.sleep(0.5)
        shot = capture_screen()
        area = find_board(shot)
        board = recognize_board(shot, area, templates)

        # 3. Solve
        moves, score = solve_with_go(board)
        print(f"Attempt {attempt:4d} | score = {score}")

        if score >= target:
            print(f"\nTarget {target} reached! Executing {len(moves)} moves...")
            cw = area["width"]  / COLS
            ch = area["height"] / GRID_ROWS
            current_board = board
            for i, move in enumerate(moves, 1):
                tight = tighten_move(move, current_board)
                perform_drag(tight, area, (cw, ch), offset)
                current_board = apply_move(move, current_board)
                if i % 10 == 0:
                    print(f"  {i}/{len(moves)} done")
            print(f"\nFinal score: {score}")
            sys.exit(0)

        # 4. Not enough — reset and retry
        if not click_reset():
            sys.exit(1)

if __name__ == "__main__":
    main()
