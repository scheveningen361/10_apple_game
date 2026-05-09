import random
import time
import sys
import math
import joblib
import numpy as np
from tqdm import tqdm

ROWS = 10
COLS = 17
TARGET_SUM = 10

def calculate_summed_area_table(board):
    """
    Calculates the summed area table for a given board (numbers or counts).
    This allows O(1) time complexity for calculating the sum of any rectangular region.
    """
    sat = [[0] * (COLS + 1) for _ in range(ROWS + 1)]
    for r in range(ROWS):
        for c in range(COLS):
            sat[r + 1][c + 1] = board[r][c] + sat[r][c + 1] + sat[r + 1][c] - sat[r][c]
    return sat

def get_rect_sum(sat, r1, c1, r2, c2):
    """Returns the sum of a rectangular region using the summed area table."""
    return sat[r2 + 1][c2 + 1] - sat[r1][c2 + 1] - sat[r2 + 1][c1] + sat[r1][c1]

def get_distance_from_center(move):
    """Calculates the distance of a move's center from the board's center."""
    r1, c1, r2, c2 = move
    board_center_r = (ROWS - 1) / 2
    board_center_c = (COLS - 1) / 2
    move_center_r = (r1 + r2) / 2
    move_center_c = (c1 + c2) / 2
    return math.sqrt((move_center_r - board_center_r)**2 + (move_center_c - board_center_c)**2)

def get_emptiness_score(board, move, radius=3):
    """
    Calculates the number of empty cells (0) in a given radius around the move.
    A higher score means more empty cells nearby.
    """
    r1, c1, r2, c2 = move
    start_r = max(0, r1 - radius)
    end_r = min(ROWS - 1, r2 + radius)
    start_c = max(0, c1 - radius)
    end_c = min(COLS - 1, c2 + radius)
    
    empty_count = 0
    for r in range(start_r, end_r + 1):
        for c in range(start_c, end_c + 1):
            if board[r][c] == 0:
                empty_count += 1
    return empty_count

def get_max_number_in_move(board, move):
    """Finds the highest number on an apple within a move's rectangle."""
    r1, c1, r2, c2 = move
    max_val = 0
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if board[r][c] > max_val:
                max_val = board[r][c]
    return max_val

def solve_min_apples_center_close(initial_board):
    """
    Greedy algorithm: min apples, tie-break with moves CLOSEST to center.
    """
    board = [row[:] for row in initial_board]
    moves_sequence = []
    total_removed = 0
    while True:
        sat_values = calculate_summed_area_table(board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_moves = find_valid_moves(board, sat_values, sat_counts)
        if not valid_moves: break

        best_move = min(valid_moves, key=lambda move: (
            get_rect_sum(sat_counts, *move),
            get_distance_from_center(move)
        ))
        
        r1, c1, r2, c2 = best_move
        total_removed += get_rect_sum(sat_counts, r1, c1, r2, c2)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                board[r][c] = 0
        moves_sequence.append(best_move)
    return moves_sequence, total_removed

def solve_min_apples_empty_bias(initial_board):
    """
    Greedy algorithm: min apples, tie-break with moves near MOST empty cells.
    """
    board = [row[:] for row in initial_board]
    moves_sequence = []
    total_removed = 0
    while True:
        sat_values = calculate_summed_area_table(board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_moves = find_valid_moves(board, sat_values, sat_counts)
        if not valid_moves: break

        best_move = min(valid_moves, key=lambda move: (
            get_rect_sum(sat_counts, *move),
            -get_emptiness_score(board, move)
        ))
        
        r1, c1, r2, c2 = best_move
        total_removed += get_rect_sum(sat_counts, r1, c1, r2, c2)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                board[r][c] = 0
        moves_sequence.append(best_move)
    return moves_sequence, total_removed

def solve_min_apples_max_number_bias(initial_board):
    """
    Greedy algorithm: min apples, tie-break with move containing the HIGHEST number.
    """
    board = [row[:] for row in initial_board]
    moves_sequence = []
    total_removed = 0
    while True:
        sat_values = calculate_summed_area_table(board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_moves = find_valid_moves(board, sat_values, sat_counts)
        if not valid_moves: break

        best_move = min(valid_moves, key=lambda move: (
            get_rect_sum(sat_counts, *move),
            -get_max_number_in_move(board, move)
        ))
        
        r1, c1, r2, c2 = best_move
        total_removed += get_rect_sum(sat_counts, r1, c1, r2, c2)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                board[r][c] = 0
        moves_sequence.append(best_move)
    return moves_sequence, total_removed

def solve_max_number_min_apples(initial_board):
    """
    Greedy algorithm: prioritizes move with HIGHEST number, tie-break with FEWEST apples.
    """
    board = [row[:] for row in initial_board]
    moves_sequence = []
    total_removed = 0
    while True:
        sat_values = calculate_summed_area_table(board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_moves = find_valid_moves(board, sat_values, sat_counts)
        if not valid_moves: break

        # Primary: max number (so use negative for min()), Tie-breaker: min apples
        best_move = min(valid_moves, key=lambda move: (
            -get_max_number_in_move(board, move),
            get_rect_sum(sat_counts, *move)
        ))
        
        r1, c1, r2, c2 = best_move
        total_removed += get_rect_sum(sat_counts, r1, c1, r2, c2)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                board[r][c] = 0
        moves_sequence.append(best_move)
    return moves_sequence, total_removed


def solve_full_simulation(initial_board):
    """
    At each step, simulates the entire rest of the game for every possible move
    and chooses the move that leads to the highest final score.
    The simulation uses the 'solve_min_apples_max_number_bias' heuristic.
    """
    board = [row[:] for row in initial_board]
    moves_sequence = []
    total_removed = 0

    while True:
        if sum(sum(row) for row in board) == 0:
            break

        sat_values = calculate_summed_area_table(board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_moves = find_valid_moves(board, sat_values, sat_counts)

        if not valid_moves:
            break

        best_move = None
        max_score = -1
        best_move_candidates = []

        # Use tqdm to show progress for simulating moves at each step
        for move in valid_moves:
            # Simulate the full game for this move and get the final score
            score = evaluate_full_game_score(board, move)
            if score > max_score:
                max_score = score
                best_move_candidates = [move]
            elif score == max_score:
                best_move_candidates.append(move)
        
        if not best_move_candidates:
            break
        
        # Tie-breaking: choose the best move from candidates using the simple heuristic
        best_move = min(best_move_candidates, key=lambda move: (
            get_rect_sum(sat_counts, *move),
            -get_max_number_in_move(board, move)
        ))

        # Make the chosen move on the actual board
        r1, c1, r2, c2 = best_move
        
        current_board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        current_sat_counts = calculate_summed_area_table(current_board_counts)
        removed_now = get_rect_sum(current_sat_counts, r1, c1, r2, c2)
        
        total_removed += removed_now

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                board[r][c] = 0
        moves_sequence.append(best_move)

    return moves_sequence, total_removed

def evaluate_full_game_score(initial_board, initial_move):
    """
    Evaluates the total score achievable from a given board state after making an initial move,
    by simulating the rest of the game using the 'solve_min_apples_max_number_bias' heuristic.
    """
    board = [row[:] for row in initial_board]
    
    # Apply the initial move
    r1, c1, r2, c2 = initial_move
    
    temp_board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
    temp_sat_counts = calculate_summed_area_table(temp_board_counts)
    score = get_rect_sum(temp_sat_counts, r1, c1, r2, c2)
    
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            board[r][c] = 0
            
    # Simulate the rest of the game from the new board state
    _, rest_of_game_score = solve_min_apples_max_number_bias(board)
    
    return score + rest_of_game_score


def find_valid_moves(board, sat_values, sat_counts):
    """
    Finds all valid rectangles (moves) that sum to 10 on the current board.
    """
    moves = []
    for r1 in range(ROWS):
        for c1 in range(COLS):
            for r2 in range(r1, ROWS):
                for c2 in range(c1, COLS):
                    current_sum = get_rect_sum(sat_values, r1, c1, r2, c2)
                    apples_in_rect = get_rect_sum(sat_counts, r1, c1, r2, c2)

                    if current_sum == TARGET_SUM and apples_in_rect >= 2:
                        moves.append((r1, c1, r2, c2))
    return moves

def print_board(board):
    """
    Prints the board to the console.
    """
    for row in board:
        print(" ".join(f"{num:2}" for num in row))

if __name__ == "__main__":
    NUM_TESTS = 100

    # --- Configuration ---
    # Add the solver functions you want to compare here.
    # 'name': The display name for the results.
    # 'func': The solver function to call.
    # 'args': A list of additional arguments to pass to the function (e.g., [depth]).
    solvers_to_test = [
        {
            "name": "Min Apples (Max Number Bias)",
            "func": solve_min_apples_max_number_bias,
            "args": []
        },
    ]
    # --- End Configuration ---

    results = {config["name"]: {"time": [], "score": []} for config in solvers_to_test}

    # Main testing loop
    for i in tqdm(range(NUM_TESTS), desc="Testing Solvers"):
        # Generate a new random board for each test run
        initial_board = [[random.randint(1, 9) for _ in range(COLS)] for _ in range(ROWS)]

        for config in solvers_to_test:
            name = config["name"]
            solver_func = config["func"]
            args = config.get("args", [])

            board_copy = [row[:] for row in initial_board]
            start_time = time.time()
            
            # Call the solver function with the board and any additional arguments
            _, total_removed = solver_func(board_copy, *args)
            
            end_time = time.time()
            
            results[name]["time"].append(end_time - start_time)
            results[name]["score"].append(total_removed)

    # Print the final results
    print("\n--- Final Results Summary ---", flush=True)
    for name, data in results.items():
        avg_time = sum(data["time"]) / NUM_TESTS if NUM_TESTS > 0 else 0
        avg_score = sum(data["score"]) / NUM_TESTS if NUM_TESTS > 0 else 0
        max_score = max(data["score"]) if data["score"] else 0
        print(f"{name}:")
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Max Score: {max_score}")
        print(f"  Average Time: {avg_time:.4f}s\n", flush=True)
