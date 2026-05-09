"""
Utility functions for RL solver
"""

import random
import numpy as np
import sys
import os

# Add parent directory to path to import apple_solver
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apple_solver import (
    ROWS, COLS, TARGET_SUM,
    calculate_summed_area_table,
    get_rect_sum,
    find_valid_moves
)

def generate_random_board():
    """Generate a random board with numbers 1-9."""
    return [[random.randint(1, 9) for _ in range(COLS)] for _ in range(ROWS)]

def extract_features(board):
    """
    Extract feature vector from board state.
    Returns a feature vector with:
    - Total apples count
    - Board density (apples / total cells)
    - Number of valid moves
    - Average apple value
    - Max apple value
    - Min apple value
    - Number distribution (count of each number 1-9)
    """
    features = []
    
    # Count apples and calculate statistics
    apple_count = 0
    total_sum = 0
    max_val = 0
    min_val = 10
    number_distribution = [0] * 9  # Count for numbers 1-9
    
    for row in board:
        for cell in row:
            if cell > 0:
                apple_count += 1
                total_sum += cell
                max_val = max(max_val, cell)
                min_val = min(min_val, cell)
                if 1 <= cell <= 9:
                    number_distribution[cell - 1] += 1
    
    # Basic features
    features.append(apple_count)  # Total apples
    features.append(apple_count / (ROWS * COLS))  # Density
    features.append(total_sum / apple_count if apple_count > 0 else 0)  # Average value
    features.append(max_val)
    features.append(min_val if min_val < 10 else 0)
    
    # Number distribution (normalized)
    if apple_count > 0:
        features.extend([count / apple_count for count in number_distribution])
    else:
        features.extend([0] * 9)
    
    # Calculate valid moves count
    sat_values = calculate_summed_area_table(board)
    board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board]
    sat_counts = calculate_summed_area_table(board_counts)
    valid_moves = find_valid_moves(board, sat_values, sat_counts)
    features.append(len(valid_moves))  # Number of valid moves
    
    return np.array(features, dtype=np.float32)

def board_to_vector(board):
    """
    Convert board to a flat vector representation.
    Returns a 1D array of size ROWS * COLS.
    """
    return np.array([cell for row in board for cell in row], dtype=np.float32)

def get_state_representation(board):
    """
    Get combined state representation: board vector + features.
    Returns a numpy array of shape (ROWS * COLS + feature_dim,)
    """
    board_vec = board_to_vector(board)
    features = extract_features(board)
    return np.concatenate([board_vec, features])

def save_model(model, path, episode=None, score=None):
    """Save model to file."""
    import torch
    save_dict = {
        'model_state_dict': model.state_dict(),
        'episode': episode,
        'score': score
    }
    torch.save(save_dict, path)

def load_model(model, path):
    """Load model from file."""
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('episode', 0), checkpoint.get('score', 0)

