"""
Test performance of a trained RL model
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_solver.trainer import PPOTrainer
from rl_solver.environment import AppleGameEnv
from rl_solver.utils import generate_random_board
from apple_solver import solve_min_apples_max_number_bias

def test_model_performance(model_path, num_tests=100):
    """
    Test model performance and compare with baseline.
    
    Args:
        model_path: Path to the model file
        num_tests: Number of test games
    """
    print("=" * 60)
    print(f"Testing Model: {model_path}")
    print("=" * 60)
    
    # Load model
    print("\n[1] Loading model...")
    trainer = PPOTrainer(state_dim=185, hidden_dim=128)
    try:
        episode, score = trainer.load_checkpoint(model_path)
        print(f"[OK] Model loaded from episode {episode}, score: {score}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Test performance
    print(f"\n[2] Testing on {num_tests} random boards...")
    print("This may take a while...\n")
    
    rl_scores = []
    baseline_scores = []
    differences = []
    test_times = []
    
    start_time = time.time()
    
    for i in range(num_tests):
        # Generate random board
        board = generate_random_board()
        
        # Test RL model
        env = AppleGameEnv(board)
        state = env.reset()
        rl_start = time.time()
        
        while not env.done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            action, _, _, _ = trainer.model.get_action(state, valid_actions, deterministic=True)
            if action is None:
                break
            
            state, _, done, _ = env.step(action)
            if done:
                break
        
        rl_time = time.time() - rl_start
        rl_score = env.get_score()
        baseline_score = env.get_baseline_score()
        diff = rl_score - baseline_score
        
        rl_scores.append(rl_score)
        baseline_scores.append(baseline_score)
        differences.append(diff)
        test_times.append(rl_time)
        
        # Progress update every 10 tests
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (num_tests - i - 1)
            print(f"  Progress: {i+1}/{num_tests} ({100*(i+1)/num_tests:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | "
                  f"Current diff: {diff:+d}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    avg_rl_score = sum(rl_scores) / len(rl_scores)
    avg_baseline_score = sum(baseline_scores) / len(baseline_scores)
    avg_diff = sum(differences) / len(differences)
    
    min_diff = min(differences)
    max_diff = max(differences)
    
    # Count wins/losses
    wins = sum(1 for d in differences if d > 0)
    losses = sum(1 for d in differences if d < 0)
    ties = sum(1 for d in differences if d == 0)
    
    win_rate = wins / num_tests * 100
    
    # Calculate improvement percentage
    improvement_pct = (avg_diff / avg_baseline_score * 100) if avg_baseline_score > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    print(f"\nTest Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Number of tests: {num_tests}")
    print(f"  Total test time: {total_time:.2f} seconds")
    print(f"  Average time per game: {total_time/num_tests:.3f} seconds")
    
    print(f"\nScore Statistics:")
    print(f"  RL Model Average Score: {avg_rl_score:.2f}")
    print(f"  Baseline Average Score: {avg_baseline_score:.2f}")
    print(f"  Average Difference: {avg_diff:+.2f} points")
    print(f"  Improvement: {improvement_pct:+.2f}%")
    
    print(f"\nDifference Range:")
    print(f"  Minimum difference: {min_diff:+d} points")
    print(f"  Maximum difference: {max_diff:+d} points")
    
    print(f"\nWin/Loss Statistics:")
    print(f"  Wins (RL > Baseline): {wins} ({win_rate:.1f}%)")
    print(f"  Losses (RL < Baseline): {losses} ({losses/num_tests*100:.1f}%)")
    print(f"  Ties (RL = Baseline): {ties} ({ties/num_tests*100:.1f}%)")
    
    # Score distribution
    print(f"\nScore Distribution:")
    print(f"  RL Model - Min: {min(rl_scores)}, Max: {max(rl_scores)}, "
          f"Std: {sum((s - avg_rl_score)**2 for s in rl_scores) / len(rl_scores) ** 0.5:.2f}")
    print(f"  Baseline - Min: {min(baseline_scores)}, Max: {max(baseline_scores)}, "
          f"Std: {sum((s - avg_baseline_score)**2 for s in baseline_scores) / len(baseline_scores) ** 0.5:.2f}")
    
    # Performance summary
    print(f"\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    
    if avg_diff > 5:
        print("  [EXCELLENT] RL model significantly outperforms baseline!")
    elif avg_diff > 2:
        print("  [GOOD] RL model outperforms baseline.")
    elif avg_diff > 0:
        print("  [FAIR] RL model slightly outperforms baseline.")
    elif avg_diff > -2:
        print("  [POOR] RL model performs similarly to baseline.")
    else:
        print("  [NEEDS IMPROVEMENT] RL model underperforms baseline.")
    
    print(f"\n  Average improvement: {avg_diff:+.2f} points ({improvement_pct:+.2f}%)")
    print(f"  Win rate: {win_rate:.1f}%")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    model_path = "rl_solver/saved_models/9h_model.pt"
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("\nAvailable models:")
        models_dir = "rl_solver/saved_models"
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith(".pt"):
                    print(f"  - {os.path.join(models_dir, f)}")
        sys.exit(1)
    
    # Test with 100 games
    test_model_performance(model_path, num_tests=100)

