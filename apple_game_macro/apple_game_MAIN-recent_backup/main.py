

import os
import mss
from PIL import Image
import cv2
import numpy as np
import json
import sys
import time
from tqdm import tqdm
import re
import pyautogui
import joblib

# --- Global Variables and Constants ---
PLAY_BUTTON_IMAGE = "play_button.png"
RESET_BUTTON_IMAGE = "reset_button.png"
CONFIG_FILE = "config.json"
TEMPLATES_DIR = "templates"
OUTPUT_DIR = "output_images"
GAME_BOARD_TEMPLATE_FILE = "game_board_template.png"
SCORE_LOG_FILE = "score_log.txt"
GRID_ROWS = 10
COLS = 17
TARGET_SCORE = 170

# --- Algorithm Imports ---
from apple_solver import print_board as print_solver_board
from apple_solver import solve_min_apples_max_number_bias as solve_optimal_moves

# --- Existing Functions ---
def load_templates():
    templates = {}
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)
        print(f"Warning: '{TEMPLATES_DIR}' folder not found, created. Please add template images.")
    for i in range(1, 10):
        template_path = os.path.join(TEMPLATES_DIR, f"{i}.png")
        if os.path.exists(template_path):
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_img is not None:
                templates[i] = template_img
    return templates

def calibrate_game_board_template():
    print("\n--- Game Board Template Setup ---")
    template_path = input("Game board template image file path: ")
    try:
        template_img_pil = Image.open(template_path).convert("RGB")
        template_img_pil.save(GAME_BOARD_TEMPLATE_FILE)
        config = {"game_board_template_width": template_img_pil.width, "game_board_template_height": template_img_pil.height}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        print(f"Game board template saved to '{GAME_BOARD_TEMPLATE_FILE}'.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def find_game_board_in_screenshot(full_screenshot_img_cv):
    if not os.path.exists(GAME_BOARD_TEMPLATE_FILE):
        print(f"오류: '{GAME_BOARD_TEMPLATE_FILE}'을 찾을 수 없습니다. --calibrate 옵션으로 설정해주세요.")
        sys.exit(1)
    template_img_cv = cv2.imread(GAME_BOARD_TEMPLATE_FILE, cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(full_screenshot_img_cv, template_img_cv, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= 0.8:
        return {"left": max_loc[0], "top": max_loc[1], "width": template_img_cv.shape[1], "height": template_img_cv.shape[0]}
    else:
        print(f"오류: 게임 보드를 찾을 수 없습니다 (매칭 점수: {max_val:.2f}).")
        sys.exit(1)

def recognize_board_from_image(full_screenshot_img_cv, capture_area, templates):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    cropped_img_cv = full_screenshot_img_cv[capture_area["top"]:capture_area["top"] + capture_area["height"],
                                            capture_area["left"]:capture_area["left"] + capture_area["width"]]
    cv2.imwrite("capture.png", cropped_img_cv)
    game_board = [[0] * COLS for _ in range(GRID_ROWS)]
    cell_width = cropped_img_cv.shape[1] / COLS
    cell_height = cropped_img_cv.shape[0] / GRID_ROWS
    for r in range(GRID_ROWS):
        for c in range(COLS):
            cell_img = cropped_img_cv[int(r*cell_height):int((r+1)*cell_height), int(c*cell_width):int((c+1)*cell_width)]
            best_match_score = -1
            recognized_number = 0
            for number, template in templates.items():
                if cell_img.shape[0] < template.shape[0] or cell_img.shape[1] < template.shape[1]:
                    continue
                res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_match_score:
                    best_match_score = max_val
                    recognized_number = number
            
            game_board[r][c] = recognized_number
    return game_board

def save_board_to_file(board, filename):
    with open(filename, 'w') as f:
        for row in board:
            f.write(" ".join(map(str, row)) + "\n")

def perform_mouse_drag(move, game_board_offset, cell_dims, mouse_offset):
    r1, c1, r2, c2 = move
    start_x = game_board_offset["left"] + (c1 * cell_dims[0]) + mouse_offset[0]
    start_y = game_board_offset["top"] + (r1 * cell_dims[1]) + mouse_offset[1]
    end_x = game_board_offset["left"] + ((c2 + 1) * cell_dims[0]) + mouse_offset[0]
    end_y = game_board_offset["top"] + ((r2 + 1) * cell_dims[1]) + mouse_offset[1]
    pyautogui.moveTo(start_x, start_y, duration=0.1)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_x, end_y, duration=0.1)
    pyautogui.moveRel(10, 0, duration=0.03)
    pyautogui.moveRel(-20, 0, duration=0.03)
    pyautogui.moveRel(10, 0, duration=0.03)
    time.sleep(0.1)
    pyautogui.mouseUp()
    time.sleep(0.3)

def calibrate_mouse_offset(game_board_area):
    print("\n--- 마우스 오프셋 보정 ---")
    input("게임 보드의 가장 왼쪽 위 모서리로 마우스를 이동하고 Enter를 누르세요...")
    actual_mouse_x, actual_mouse_y = pyautogui.position()
    offset_x = actual_mouse_x - game_board_area["left"]
    offset_y = actual_mouse_y - game_board_area["top"]
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    config["mouse_offset_x"] = offset_x
    config["mouse_offset_y"] = offset_y
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)
    print(f"마우스 오프셋 저장 완료: x={offset_x}, y={offset_y}")

def find_and_click_button(image_path, button_name):
    print(f"\n{button_name} 버튼을 찾는 중...")
    try:
        button_location = pyautogui.locateOnScreen(image_path, confidence=0.9)
        if button_location:
            print(f"{button_name} 버튼을 찾았습니다: {button_location}")
            pyautogui.click(button_location)
            print(f"{button_name} 버튼을 클릭했습니다.")
            time.sleep(1)
            return True
        else:
            print(f"{button_name} 버튼을 찾을 수 없습니다.")
            return False
    except Exception as e:
        print(f"{button_name} 버튼을 찾는 중 오류 발생: {e}")
        return False

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    templates = load_templates()
    if "--calibrate" in sys.argv or not os.path.exists(GAME_BOARD_TEMPLATE_FILE):
        calibrate_game_board_template()

    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    mouse_offset_x = config.get("mouse_offset_x", 0)
    mouse_offset_y = config.get("mouse_offset_y", 0)

    highest_score = 150
    try:
        highest_score=int(input("highest score: "))
    except:
        print("highest score = 150")
    try:
        TARGET_SCORE=int(input("target score: "))
    except:
        print("target score = 170")
    recording=1
    try:
        recording=int(input("recording(0 or 1): "))
    except:
        print("recording = True")

    time.sleep(3)

    program_start_time = time.time()


    def log_score_event(score, event_type):
        elapsed_time = time.time() - program_start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        log_message = f"[{time_str}] {event_type}: {score}점"
        with open(SCORE_LOG_FILE, 'a') as f:
            f.write(log_message + "\n")
        print(f"로그 기록: {log_message}")

    print("\n프로그램 시작. 플레이 버튼을 클릭하여 게임을 시작합니다.")
    find_and_click_button(PLAY_BUTTON_IMAGE, "플레이")
    
    while True:
        print("\n--- 새 게임 시작 ---")
        print("1초 후 전체 화면 캡처를 시작합니다...")
        time.sleep(1)
        with mss.mss() as sct:
            sct_img = sct.grab(sct.monitors[0])
            full_screenshot_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            full_screenshot_cv = cv2.cvtColor(np.array(full_screenshot_pil), cv2.COLOR_RGB2GRAY)
        
        game_board_area = find_game_board_in_screenshot(full_screenshot_cv)
        print(f"게임 보드 영역: {game_board_area}")

        if "--calibrate-mouse" in sys.argv or ("mouse_offset_x" not in config or "mouse_offset_y" not in config):
            calibrate_mouse_offset(game_board_area)
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            mouse_offset_x = config.get("mouse_offset_x", 0)
            mouse_offset_y = config.get("mouse_offset_y", 0)

        current_board = recognize_board_from_image(full_screenshot_cv, game_board_area, templates)
        print("\n--- 인식된 게임 보드 ---")
        print_solver_board(current_board)
        
        optimal_moves, total_removed = solve_optimal_moves(current_board)
        print(f"\n예상 점수: {total_removed}점 (현재 최고 점수: {highest_score}점)")

        if total_removed > highest_score:
            print(f"새로운 최고 점수 달성! {total_removed}점 (이전: {highest_score}점)")
            log_score_event(total_removed, "NEW_HIGH_SCORE")
            highest_score = total_removed

            if not optimal_moves:
                print("실행할 움직임이 없습니다.")
            else:
                print(f"총 {len(optimal_moves)}개의 움직임으로 자동 플레이 시작...")
                cell_width_px = game_board_area["width"] / COLS
                cell_height_px = game_board_area["height"] / GRID_ROWS
                for i, move in enumerate(optimal_moves):
                    perform_mouse_drag(move, game_board_area, (cell_width_px, cell_height_px), (mouse_offset_x, mouse_offset_y))
                print("\n자동 플레이 완료.")

            # 게임 플레이 종료 후 녹화 단축키 누르기
            time.sleep(1)
            if recording:
                print("3초 후, 지난 플레이를 녹화하기 위해 Win+Alt+G 단축키를 누릅니다...")
                time.sleep(2)
                pyautogui.hotkey('win', 'alt', 'g')
                print("녹화 단축키를 눌렀습니다.")

            if total_removed >= TARGET_SCORE:
                print("목표 점수 달성! 프로그램 종료.")
                sys.exit(0)
            
            print("리셋을 준비합니다...")

        else:
            print(f"예상 점수가 최고 점수보다 낮으므로 리셋합니다.")

        find_and_click_button(RESET_BUTTON_IMAGE, "리셋")
        find_and_click_button(PLAY_BUTTON_IMAGE, "플레이")
