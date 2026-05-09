"""
Game Environment for RL training
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apple_solver import (
    ROWS, COLS, TARGET_SUM,
    calculate_summed_area_table,
    get_rect_sum,
    find_valid_moves,
    solve_min_apples_max_number_bias
)
from rl_solver.utils import (
    generate_random_board,
    get_state_representation
)

class AppleGameEnv:
    """
    Apple Game Environment for Reinforcement Learning.
    
    Reward: Only final reward at game end (difference from baseline score)
    No immediate reward per move to avoid greedy behavior.
    """
    
    def __init__(self, initial_board=None):
        """
        Initialize environment.
        
        Args:
            initial_board: Optional initial board. If None, generates random board.
        """
        self.initial_board = initial_board if initial_board is not None else generate_random_board()
        self.board = [row[:] for row in self.initial_board]
        self.done = False
        self.total_removed = 0
        self.moves_made = []
        
        # Calculate baseline score for this board
        baseline_board = [row[:] for row in self.initial_board]
        _, self.baseline_score = solve_min_apples_max_number_bias(baseline_board)
    
    def reset(self, new_board=None):
        """
        Reset environment to initial state or new board.
        
        Args:
            new_board: Optional new board. If None, uses initial board.
        
        Returns:
            state: Current state representation
        """
        if new_board is not None:
            self.initial_board = new_board
        
        self.board = [row[:] for row in self.initial_board]
        self.done = False
        self.total_removed = 0
        self.moves_made = []
        
        # Recalculate baseline for new board
        baseline_board = [row[:] for row in self.initial_board]
        _, self.baseline_score = solve_min_apples_max_number_bias(baseline_board)
        
        return self.get_state()
    
    def get_state(self):
        """Get current state representation."""
        return get_state_representation(self.board)
    
    def get_valid_actions(self):
        """
        Get list of valid actions (moves) for current board state.
        
        Returns:
            valid_actions: List of valid moves (r1, c1, r2, c2)
        """
        sat_values = calculate_summed_area_table(self.board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in self.board]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_moves = find_valid_moves(self.board, sat_values, sat_counts)
        return valid_moves
    
    def step(self, action):
        """
        Execute an action (move).
        
        Args:
            action: Tuple (r1, c1, r2, c2) representing the move
        
        Returns:
            state: New state after action
            reward: Reward (0 during game, final reward at end)
            done: Whether game is finished
            info: Additional information
        """
        if self.done:
            return self.get_state(), 0, True, {}
        
        r1, c1, r2, c2 = action
        
        # Check if action is valid
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            # Invalid action - end game with negative reward
            self.done = True
            reward = -self.baseline_score  # Heavy penalty for invalid action
            return self.get_state(), reward, True, {'invalid_action': True}
        
        # Apply move
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in self.board]
        sat_counts = calculate_summed_area_table(board_counts)
        apples_removed = get_rect_sum(sat_counts, r1, c1, r2, c2)
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.board[r][c] = 0
        
        self.total_removed += apples_removed
        self.moves_made.append(action)
        
        # Check if game is done (no more valid moves)
        sat_values = calculate_summed_area_table(self.board)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in self.board]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_actions = find_valid_moves(self.board, sat_values, sat_counts)
        
        if not valid_actions:
            self.done = True
            # Final reward: difference from baseline
            reward = self.total_removed - self.baseline_score
        else:
            # No immediate reward during game
            reward = 0
        
        info = {
            'apples_removed': apples_removed,
            'total_removed': self.total_removed,
            'baseline_score': self.baseline_score,
            'done': self.done
        }
        
        return self.get_state(), reward, self.done, info
    
    def get_score(self):
        """Get current total score."""
        return self.total_removed
    
    def get_baseline_score(self):
        """Get baseline score for comparison."""
        return self.baseline_score

