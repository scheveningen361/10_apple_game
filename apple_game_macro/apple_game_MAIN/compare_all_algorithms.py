"""
Compare Random Play, Baseline Heuristic, and RL Model
Comprehensive statistics comparison
"""

import sys
import os
import time
import random
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_solver.trainer import PPOTrainer
from rl_solver.environment import AppleGameEnv
from rl_solver.utils import generate_random_board
from apple_solver import solve_min_apples_max_number_bias, find_valid_moves, calculate_summed_area_table, get_rect_sum

def random_play(board):
    """
    Random play: randomly select valid moves until no moves available.
    
    Returns:
        score: Total apples removed
        moves_count: Number of moves made
    """
    board_copy = [row[:] for row in board]
    total_removed = 0
    moves_count = 0
    
    while True:
        sat_values = calculate_summed_area_table(board_copy)
        board_counts = [[1 if cell > 0 else 0 for cell in row] for row in board_copy]
        sat_counts = calculate_summed_area_table(board_counts)
        valid_moves = find_valid_moves(board_copy, sat_values, sat_counts)
        
        if not valid_moves:
            break
        
        # Randomly select a move
        move = random.choice(valid_moves)
        r1, c1, r2, c2 = move
        
        apples_removed = get_rect_sum(sat_counts, r1, c1, r2, c2)
        total_removed += apples_removed
        moves_count += 1
        
        # Apply move
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                board_copy[r][c] = 0
    
    return total_removed, moves_count

def baseline_play(board):
    """
    Baseline heuristic: solve_min_apples_max_number_bias.
    
    Returns:
        score: Total apples removed
        moves_count: Number of moves made
    """
    board_copy = [row[:] for row in board]
    moves_sequence, total_removed = solve_min_apples_max_number_bias(board_copy)
    return total_removed, len(moves_sequence)

def rl_model_play(board, trainer):
    """
    RL model play: use trained model with deterministic policy.
    
    Returns:
        score: Total apples removed
        moves_count: Number of moves made
    """
    env = AppleGameEnv(board)
    state = env.reset()
    moves_count = 0
    
    while not env.done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        
        action, _, _, _ = trainer.model.get_action(state, valid_actions, deterministic=True)
        if action is None:
            break
        
        state, _, done, _ = env.step(action)
        moves_count += 1
        if done:
            break
    
    return env.get_score(), moves_count

def compare_all_algorithms(model_path, num_tests=100):
    """
    Compare all three algorithms comprehensively.
    
    Args:
        model_path: Path to RL model file
        num_tests: Number of test games
    """
    print("=" * 80)
    print("Comprehensive Algorithm Comparison")
    print("=" * 80)
    print(f"\nAlgorithms:")
    print("  1. Random Play")
    print("  2. Baseline Heuristic (solve_min_apples_max_number_bias)")
    print("  3. RL Model (9h_model.pt)")
    print(f"\nNumber of test games: {num_tests}")
    print("\n" + "=" * 80)
    
    # Load RL model
    print("\n[Loading] RL Model...")
    trainer = PPOTrainer(state_dim=185, hidden_dim=128)
    try:
        episode, score = trainer.load_checkpoint(model_path)
        print(f"[OK] Model loaded from episode {episode}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Test results storage
    random_scores = []
    baseline_scores = []
    rl_scores = []
    
    random_moves = []
    baseline_moves = []
    rl_moves = []
    
    random_times = []
    baseline_times = []
    rl_times = []
    
    # Detailed statistics
    score_diffs_rl_vs_baseline = []
    score_diffs_rl_vs_random = []
    score_diffs_baseline_vs_random = []
    
    win_counts = {
        'rl_vs_baseline': {'rl': 0, 'baseline': 0, 'tie': 0},
        'rl_vs_random': {'rl': 0, 'random': 0, 'tie': 0},
        'baseline_vs_random': {'baseline': 0, 'random': 0, 'tie': 0}
    }
    
    print("\n[Testing] Running tests...")
    print("Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for i in range(num_tests):
        # Generate random board
        board = generate_random_board()
        
        # Test Random Play
        t0 = time.time()
        random_score, random_move_count = random_play(board)
        random_times.append(time.time() - t0)
        random_scores.append(random_score)
        random_moves.append(random_move_count)
        
        # Test Baseline
        t0 = time.time()
        baseline_score, baseline_move_count = baseline_play(board)
        baseline_times.append(time.time() - t0)
        baseline_scores.append(baseline_score)
        baseline_moves.append(baseline_move_count)
        
        # Test RL Model
        t0 = time.time()
        rl_score, rl_move_count = rl_model_play(board, trainer)
        rl_times.append(time.time() - t0)
        rl_scores.append(rl_score)
        rl_moves.append(rl_move_count)
        
        # Calculate differences
        diff_rl_baseline = rl_score - baseline_score
        diff_rl_random = rl_score - random_score
        diff_baseline_random = baseline_score - random_score
        
        score_diffs_rl_vs_baseline.append(diff_rl_baseline)
        score_diffs_rl_vs_random.append(diff_rl_random)
        score_diffs_baseline_vs_random.append(diff_baseline_random)
        
        # Count wins
        if diff_rl_baseline > 0:
            win_counts['rl_vs_baseline']['rl'] += 1
        elif diff_rl_baseline < 0:
            win_counts['rl_vs_baseline']['baseline'] += 1
        else:
            win_counts['rl_vs_baseline']['tie'] += 1
        
        if diff_rl_random > 0:
            win_counts['rl_vs_random']['rl'] += 1
        elif diff_rl_random < 0:
            win_counts['rl_vs_random']['random'] += 1
        else:
            win_counts['rl_vs_random']['tie'] += 1
        
        if diff_baseline_random > 0:
            win_counts['baseline_vs_random']['baseline'] += 1
        elif diff_baseline_random < 0:
            win_counts['baseline_vs_random']['random'] += 1
        else:
            win_counts['baseline_vs_random']['tie'] += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"{i+1} ", end="", flush=True)
    
    total_time = time.time() - start_time
    
    print(f"\n\n[Completed] Total time: {total_time:.2f} seconds")
    print("=" * 80)
    
    # Calculate statistics
    def calc_stats(scores, moves, times, name):
        """Calculate comprehensive statistics."""
        stats = {}
        stats['name'] = name
        stats['scores'] = scores
        stats['moves'] = moves
        stats['times'] = times
        
        # Score statistics
        stats['score_mean'] = np.mean(scores)
        stats['score_median'] = np.median(scores)
        stats['score_std'] = np.std(scores)
        stats['score_min'] = np.min(scores)
        stats['score_max'] = np.max(scores)
        stats['score_q25'] = np.percentile(scores, 25)
        stats['score_q75'] = np.percentile(scores, 75)
        
        # Move statistics
        stats['move_mean'] = np.mean(moves)
        stats['move_median'] = np.median(moves)
        stats['move_std'] = np.std(moves)
        
        # Time statistics
        stats['time_mean'] = np.mean(times)
        stats['time_total'] = np.sum(times)
        
        return stats
    
    random_stats = calc_stats(random_scores, random_moves, random_times, "Random Play")
    baseline_stats = calc_stats(baseline_scores, baseline_moves, baseline_times, "Baseline Heuristic")
    rl_stats = calc_stats(rl_scores, rl_moves, rl_times, "RL Model")
    
    # Print comprehensive results
    print("\n" + "=" * 80)
    print("SCORE STATISTICS")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Random':>15} {'Baseline':>15} {'RL Model':>15}")
    print("-" * 80)
    print(f"{'Mean Score':<25} {random_stats['score_mean']:>15.2f} {baseline_stats['score_mean']:>15.2f} {rl_stats['score_mean']:>15.2f}")
    print(f"{'Median Score':<25} {random_stats['score_median']:>15.2f} {baseline_stats['score_median']:>15.2f} {rl_stats['score_median']:>15.2f}")
    print(f"{'Std Deviation':<25} {random_stats['score_std']:>15.2f} {baseline_stats['score_std']:>15.2f} {rl_stats['score_std']:>15.2f}")
    print(f"{'Min Score':<25} {random_stats['score_min']:>15.0f} {baseline_stats['score_min']:>15.0f} {rl_stats['score_min']:>15.0f}")
    print(f"{'Max Score':<25} {random_stats['score_max']:>15.0f} {baseline_stats['score_max']:>15.0f} {rl_stats['score_max']:>15.0f}")
    print(f"{'Q1 (25th percentile)':<25} {random_stats['score_q25']:>15.2f} {baseline_stats['score_q25']:>15.2f} {rl_stats['score_q25']:>15.2f}")
    print(f"{'Q3 (75th percentile)':<25} {random_stats['score_q75']:>15.2f} {baseline_stats['score_q75']:>15.2f} {rl_stats['score_q75']:>15.2f}")
    
    print("\n" + "=" * 80)
    print("MOVE COUNT STATISTICS")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Random':>15} {'Baseline':>15} {'RL Model':>15}")
    print("-" * 80)
    print(f"{'Mean Moves':<25} {random_stats['move_mean']:>15.2f} {baseline_stats['move_mean']:>15.2f} {rl_stats['move_mean']:>15.2f}")
    print(f"{'Median Moves':<25} {random_stats['move_median']:>15.2f} {baseline_stats['move_median']:>15.2f} {rl_stats['move_median']:>15.2f}")
    print(f"{'Std Deviation':<25} {random_stats['move_std']:>15.2f} {baseline_stats['move_std']:>15.2f} {rl_stats['move_std']:>15.2f}")
    
    print("\n" + "=" * 80)
    print("EXECUTION TIME STATISTICS")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Random':>15} {'Baseline':>15} {'RL Model':>15}")
    print("-" * 80)
    print(f"{'Mean Time (ms)':<25} {random_stats['time_mean']*1000:>15.2f} {baseline_stats['time_mean']*1000:>15.2f} {rl_stats['time_mean']*1000:>15.2f}")
    print(f"{'Total Time (s)':<25} {random_stats['time_total']:>15.2f} {baseline_stats['time_total']:>15.2f} {rl_stats['time_total']:>15.2f}")
    
    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISON")
    print("=" * 80)
    
    # RL vs Baseline
    avg_diff_rl_baseline = np.mean(score_diffs_rl_vs_baseline)
    print(f"\n[RL Model vs Baseline]")
    print(f"  Average difference: {avg_diff_rl_baseline:+.2f} points")
    print(f"  Median difference: {np.median(score_diffs_rl_vs_baseline):+.2f} points")
    print(f"  Std deviation: {np.std(score_diffs_rl_vs_baseline):.2f} points")
    print(f"  Min difference: {np.min(score_diffs_rl_vs_baseline):+.0f} points")
    print(f"  Max difference: {np.max(score_diffs_rl_vs_baseline):+.0f} points")
    print(f"  Win rate: RL={win_counts['rl_vs_baseline']['rl']} ({win_counts['rl_vs_baseline']['rl']/num_tests*100:.1f}%), "
          f"Baseline={win_counts['rl_vs_baseline']['baseline']} ({win_counts['rl_vs_baseline']['baseline']/num_tests*100:.1f}%), "
          f"Tie={win_counts['rl_vs_baseline']['tie']} ({win_counts['rl_vs_baseline']['tie']/num_tests*100:.1f}%)")
    improvement_pct = (avg_diff_rl_baseline / baseline_stats['score_mean'] * 100) if baseline_stats['score_mean'] > 0 else 0
    print(f"  Improvement: {improvement_pct:+.2f}%")
    
    # RL vs Random
    avg_diff_rl_random = np.mean(score_diffs_rl_vs_random)
    print(f"\n[RL Model vs Random]")
    print(f"  Average difference: {avg_diff_rl_random:+.2f} points")
    print(f"  Median difference: {np.median(score_diffs_rl_vs_random):+.2f} points")
    print(f"  Std deviation: {np.std(score_diffs_rl_vs_random):.2f} points")
    print(f"  Min difference: {np.min(score_diffs_rl_vs_random):+.0f} points")
    print(f"  Max difference: {np.max(score_diffs_rl_vs_random):+.0f} points")
    print(f"  Win rate: RL={win_counts['rl_vs_random']['rl']} ({win_counts['rl_vs_random']['rl']/num_tests*100:.1f}%), "
          f"Random={win_counts['rl_vs_random']['random']} ({win_counts['rl_vs_random']['random']/num_tests*100:.1f}%), "
          f"Tie={win_counts['rl_vs_random']['tie']} ({win_counts['rl_vs_random']['tie']/num_tests*100:.1f}%)")
    improvement_pct = (avg_diff_rl_random / random_stats['score_mean'] * 100) if random_stats['score_mean'] > 0 else 0
    print(f"  Improvement: {improvement_pct:+.2f}%")
    
    # Baseline vs Random
    avg_diff_baseline_random = np.mean(score_diffs_baseline_vs_random)
    print(f"\n[Baseline vs Random]")
    print(f"  Average difference: {avg_diff_baseline_random:+.2f} points")
    print(f"  Median difference: {np.median(score_diffs_baseline_vs_random):+.2f} points")
    print(f"  Std deviation: {np.std(score_diffs_baseline_vs_random):.2f} points")
    print(f"  Min difference: {np.min(score_diffs_baseline_vs_random):+.0f} points")
    print(f"  Max difference: {np.max(score_diffs_baseline_vs_random):+.0f} points")
    print(f"  Win rate: Baseline={win_counts['baseline_vs_random']['baseline']} ({win_counts['baseline_vs_random']['baseline']/num_tests*100:.1f}%), "
          f"Random={win_counts['baseline_vs_random']['random']} ({win_counts['baseline_vs_random']['random']/num_tests*100:.1f}%), "
          f"Tie={win_counts['baseline_vs_random']['tie']} ({win_counts['baseline_vs_random']['tie']/num_tests*100:.1f}%)")
    improvement_pct = (avg_diff_baseline_random / random_stats['score_mean'] * 100) if random_stats['score_mean'] > 0 else 0
    print(f"  Improvement: {improvement_pct:+.2f}%")
    
    # Ranking
    print("\n" + "=" * 80)
    print("RANKING SUMMARY")
    print("=" * 80)
    
    algorithms = [
        ("Random Play", random_stats['score_mean']),
        ("Baseline Heuristic", baseline_stats['score_mean']),
        ("RL Model", rl_stats['score_mean'])
    ]
    algorithms.sort(key=lambda x: x[1], reverse=True)
    
    print("\nRanking by Average Score:")
    for rank, (name, score) in enumerate(algorithms, 1):
        print(f"  {rank}. {name}: {score:.2f} points")
    
    # Score distribution analysis
    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    def analyze_distribution(scores, name):
        """Analyze score distribution."""
        print(f"\n{name}:")
        bins = [0, 50, 75, 100, 125, 150, 200]
        bin_labels = ["0-50", "51-75", "76-100", "101-125", "126-150", "151+"]
        hist, _ = np.histogram(scores, bins=bins)
        for label, count in zip(bin_labels, hist):
            pct = count / len(scores) * 100
            print(f"  {label:8s}: {count:3d} games ({pct:5.1f}%)")
    
    analyze_distribution(random_scores, "Random Play")
    analyze_distribution(baseline_scores, "Baseline Heuristic")
    analyze_distribution(rl_scores, "RL Model")
    
    # Consistency analysis
    print("\n" + "=" * 80)
    print("CONSISTENCY ANALYSIS (Coefficient of Variation)")
    print("=" * 80)
    
    cv_random = (random_stats['score_std'] / random_stats['score_mean'] * 100) if random_stats['score_mean'] > 0 else 0
    cv_baseline = (baseline_stats['score_std'] / baseline_stats['score_mean'] * 100) if baseline_stats['score_mean'] > 0 else 0
    cv_rl = (rl_stats['score_std'] / rl_stats['score_mean'] * 100) if rl_stats['score_mean'] > 0 else 0
    
    print(f"\nCoefficient of Variation (lower is more consistent):")
    print(f"  Random Play: {cv_random:.2f}%")
    print(f"  Baseline: {cv_baseline:.2f}%")
    print(f"  RL Model: {cv_rl:.2f}%")
    
    # Efficiency (score per move)
    print("\n" + "=" * 80)
    print("EFFICIENCY ANALYSIS (Score per Move)")
    print("=" * 80)
    
    efficiency_random = random_stats['score_mean'] / random_stats['move_mean'] if random_stats['move_mean'] > 0 else 0
    efficiency_baseline = baseline_stats['score_mean'] / baseline_stats['move_mean'] if baseline_stats['move_mean'] > 0 else 0
    efficiency_rl = rl_stats['score_mean'] / rl_stats['move_mean'] if rl_stats['move_mean'] > 0 else 0
    
    print(f"\nAverage score per move:")
    print(f"  Random Play: {efficiency_random:.3f}")
    print(f"  Baseline: {efficiency_baseline:.3f}")
    print(f"  RL Model: {efficiency_rl:.3f}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\nPerformance Ranking:")
    for rank, (name, score) in enumerate(algorithms, 1):
        print(f"  {rank}. {name}: {score:.2f} points")
    
    print(f"\nKey Findings:")
    if avg_diff_rl_baseline > 0:
        print(f"  - RL Model outperforms Baseline by {avg_diff_rl_baseline:.2f} points on average")
    else:
        print(f"  - RL Model underperforms Baseline by {abs(avg_diff_rl_baseline):.2f} points on average")
    
    if avg_diff_rl_random > 0:
        print(f"  - RL Model outperforms Random by {avg_diff_rl_random:.2f} points on average")
    else:
        print(f"  - RL Model underperforms Random by {abs(avg_diff_rl_random):.2f} points on average")
    
    print(f"  - Baseline outperforms Random by {avg_diff_baseline_random:.2f} points on average")
    
    print(f"\nConsistency (CV):")
    print(f"  Most consistent: {min([('Random', cv_random), ('Baseline', cv_baseline), ('RL', cv_rl)], key=lambda x: x[1])[0]}")
    print(f"  Least consistent: {max([('Random', cv_random), ('Baseline', cv_baseline), ('RL', cv_rl)], key=lambda x: x[1])[0]}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    model_path = "rl_solver/saved_models/9h_model.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    compare_all_algorithms(model_path, num_tests=100)


