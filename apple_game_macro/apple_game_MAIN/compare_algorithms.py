"""
Algorithm Comparison Script
Compares solve_min_apples_max_number_bias with solve_future_potential
"""

import random
import time
from apple_solver import (
    solve_min_apples_max_number_bias,
    solve_future_potential
)

ROWS = 10
COLS = 17
NUM_GAMES = 10

def generate_random_board():
    """Generate a random board with numbers 1-9."""
    return [[random.randint(1, 9) for _ in range(COLS)] for _ in range(ROWS)]

def compare_algorithms():
    """Compare the two algorithms on identical starting boards."""
    
    score_differences = []
    time_old = []
    time_new = []
    
    print("=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    print(f"Number of games: {NUM_GAMES}\n")
    
    for game_num in range(1, NUM_GAMES + 1):
        # Generate identical starting board for both algorithms
        initial_board = generate_random_board()
        
        # Test old algorithm
        board_copy_old = [row[:] for row in initial_board]
        start_time_old = time.time()
        _, score_old = solve_min_apples_max_number_bias(board_copy_old)
        end_time_old = time.time()
        elapsed_old = end_time_old - start_time_old
        time_old.append(elapsed_old)
        
        # Test new algorithm
        board_copy_new = [row[:] for row in initial_board]
        start_time_new = time.time()
        _, score_new = solve_future_potential(board_copy_new)
        end_time_new = time.time()
        elapsed_new = end_time_new - start_time_new
        time_new.append(elapsed_new)
        
        # Calculate difference (new - old)
        difference = score_new - score_old
        score_differences.append(difference)
        
        # Print results for this game
        print(f"Game {game_num}:")
        print(f"  Old Algorithm (solve_min_apples_max_number_bias):")
        print(f"    Score: {score_old} points")
        print(f"    Time: {elapsed_old:.4f} seconds")
        print(f"  New Algorithm (solve_future_potential):")
        print(f"    Score: {score_new} points")
        print(f"    Time: {elapsed_new:.4f} seconds")
        print(f"  Difference (New - Old): {difference:+d} points")
        print()
    
    # Calculate statistics
    avg_difference = sum(score_differences) / NUM_GAMES
    min_difference = min(score_differences)
    max_difference = max(score_differences)
    
    avg_time_old = sum(time_old) / NUM_GAMES
    avg_time_new = sum(time_new) / NUM_GAMES
    
    # Print summary
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Score Difference (New - Old):")
    print(f"  Average: {avg_difference:+.2f} points")
    print(f"  Minimum: {min_difference:+d} points")
    print(f"  Maximum: {max_difference:+d} points")
    print()
    print(f"Average Execution Time:")
    print(f"  Old Algorithm: {avg_time_old:.4f} seconds")
    print(f"  New Algorithm: {avg_time_new:.4f} seconds")
    print(f"  Speed Ratio: {avg_time_new / avg_time_old:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    compare_algorithms()

