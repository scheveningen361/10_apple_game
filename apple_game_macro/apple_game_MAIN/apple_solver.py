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


def moves_overlap(move1, move2):
    """
    Check if two moves share any cells.
    Returns True if moves overlap, False otherwise.
    """
    r1_1, c1_1, r2_1, c2_1 = move1
    r1_2, c1_2, r2_2, c2_2 = move2
    
    # Check if rectangles overlap
    # Two rectangles overlap if they share at least one cell
    if r2_1 < r1_2 or r2_2 < r1_1:
        return False  # No vertical overlap
    if c2_1 < c1_2 or c2_2 < c1_1:
        return False  # No horizontal overlap
    return True  # Rectangles overlap

def get_apples_in_move(board, move, sat_counts=None):
    """
    Count the number of apples (non-zero cells) in a move.
    Uses SAT for efficiency. If sat_counts is provided, uses it instead of recalculating.
    """
    if sat_counts is None:
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)
    r1, c1, r2, c2 = move
    return get_rect_sum(sat_counts, r1, c1, r2, c2)

def find_maximum_independent_set(moves, board, sat_counts=None):
    """
    Find the combination of non-overlapping moves that maximizes total apples removed.
    Uses greedy approach: sort moves by apples/apple_ratio and select non-overlapping ones.
    Returns: maximum apples that can be removed.
    If sat_counts is provided, uses it instead of recalculating for each move.
    """
    if not moves:
        return 0
    
    # Calculate SAT once if not provided
    if sat_counts is None:
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)
    
    # Calculate apples for each move using the same SAT
    move_apples = [(move, get_apples_in_move(board, move, sat_counts)) for move in moves]
    # Sort by apples (descending) - greedy: take moves with most apples first
    move_apples.sort(key=lambda x: x[1], reverse=True)
    
    selected_moves = []
    total_apples = 0
    
    for move, apples in move_apples:
        # Check if this move overlaps with any already selected move
        overlaps = False
        for selected_move in selected_moves:
            if moves_overlap(move, selected_move):
                overlaps = True
                break
        
        if not overlaps:
            selected_moves.append(move)
            total_apples += apples
    
    return total_apples

def evaluate_move_potential(board, move):
    """
    Evaluate a move's potential by:
    1. Applying the move to get new board state (in-place modification)
    2. Finding all valid moves in new state
    3. Finding maximum independent set of those moves
    4. Returning the total apples in that set
    5. Restoring the board to original state
    """
    r1, c1, r2, c2 = move
    
    # Store original values for restoration
    original_values = []
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            original_values.append((r, c, board[r][c]))
            board[r][c] = 0
    
    # Find all valid moves in the new board state
    sat_values = calculate_summed_area_table(board)
    board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
    sat_counts = calculate_summed_area_table(board_counts)
    future_moves = find_valid_moves(board, sat_values, sat_counts)
    
    # Find maximum independent set of future moves (pass SAT to avoid recalculation)
    max_apples = find_maximum_independent_set(future_moves, board, sat_counts)
    
    # Restore original board values
    for r, c, value in original_values:
        board[r][c] = value
    
    return max_apples

def solve_future_potential(initial_board):
    """
    Main solver function that evaluates each move by its future potential.
    For each valid move, calculates the maximum apples that can be removed
    from future non-overlapping moves, and selects the move with highest potential.
    """
    board = [row[:] for row in initial_board]
    moves_sequence = []
    total_removed = 0
    
    while True:
        sat_values = calculate_summed_area_table(board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_moves = find_valid_moves(board, sat_values, sat_counts)
        
        if not valid_moves:
            break
        
        # Evaluate potential for each move
        best_move = None
        max_potential = -1
        
        for move in valid_moves:
            potential = evaluate_move_potential(board, move)
            if potential > max_potential:
                max_potential = potential
                best_move = move
        
        if best_move is None:
            break
        
        # Apply the best move
        r1, c1, r2, c2 = best_move
        apples_removed = get_rect_sum(sat_counts, r1, c1, r2, c2)
        total_removed += apples_removed
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                board[r][c] = 0
        
        moves_sequence.append(best_move)
    
    return moves_sequence, total_removed

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


def find_all_valid_moves(sat_values):
    """
    Finds all valid rectangles (moves) that sum to 10.
    No minimum apple count filter — mirrors the Go AnA algorithm exactly.
    """
    moves = []
    for r1 in range(ROWS):
        for c1 in range(COLS):
            for r2 in range(r1, ROWS):
                for c2 in range(c1, COLS):
                    if get_rect_sum(sat_values, r1, c1, r2, c2) == TARGET_SUM:
                        moves.append((r1, c1, r2, c2))
    return moves


def greedy_score(board):
    """
    Plays Greedy (min apples, max number tie-break) from the given board
    and returns the total apples removed. Used as the evaluation function in AnA.
    """
    _, score = solve_min_apples_max_number_bias(board)
    return score


def solve_ana(initial_board):
    """
    AnA (All-candidates + Greedy simulation):
    Go 구현의 playMCGreedyAllCands 와 동일한 알고리즘.

    매 스텝마다:
      1. min-count 필터 없이 합=10인 모든 직사각형을 후보로 수집
      2. 각 후보에 대해 적용 후 Greedy 점수를 계산
      3. 가장 높은 Greedy 점수를 주는 후보 선택
    """
    board = [row[:] for row in initial_board]
    moves_sequence = []
    total_removed = 0

    while True:
        sat_values = calculate_summed_area_table(board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
        sat_counts = calculate_summed_area_table(board_counts)

        candidates = find_all_valid_moves(sat_values)
        if not candidates:
            break

        best_move = None
        best_score = -1

        for move in candidates:
            # 후보 적용 후 보드 복사
            b2 = [row[:] for row in board]
            r1, c1, r2, c2 = move
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    b2[r][c] = 0

            score = greedy_score(b2)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            break

        r1, c1, r2, c2 = best_move
        removed = get_rect_sum(sat_counts, r1, c1, r2, c2)
        total_removed += removed

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                board[r][c] = 0
        moves_sequence.append(best_move)

    return moves_sequence, total_removed

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
