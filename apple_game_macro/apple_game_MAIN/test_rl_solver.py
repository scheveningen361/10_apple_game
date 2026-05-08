"""
Test script for RL Solver without GUI
Quick test to check for errors
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_solver.trainer import PPOTrainer
from rl_solver.environment import AppleGameEnv
from rl_solver.utils import generate_random_board

def test_rl_solver():
    """Quick test of RL solver components."""
    print("=" * 60)
    print("Testing RL Solver Components")
    print("=" * 60)
    
    try:
        # Test 1: Environment creation
        print("\n[Test 1] Creating environment...")
        board = generate_random_board()
        env = AppleGameEnv(board)
        state = env.reset()
        print(f"[OK] Environment created. State shape: {state.shape}")
        
        # Test 2: Get valid actions
        print("\n[Test 2] Getting valid actions...")
        valid_actions = env.get_valid_actions()
        print(f"[OK] Found {len(valid_actions)} valid actions")
        
        # Test 3: Model creation
        print("\n[Test 3] Creating model...")
        trainer = PPOTrainer(
            state_dim=185,
            hidden_dim=64,  # Smaller for faster testing
            lr=3e-4,
            save_interval=10,
            test_interval=5,
            same_board_episodes=3  # Very short for testing
        )
        print("[OK] Model created")
        
        # Test 4: Single episode training
        print("\n[Test 4] Training single episode...")
        total_reward, score, baseline = trainer.train_episode(env)
        print(f"[OK] Episode completed. Score: {score}, Baseline: {baseline}, Reward: {total_reward}")
        
        # Test 5: Multiple episodes on same board
        print("\n[Test 5] Training multiple episodes on same board...")
        env2 = AppleGameEnv(generate_random_board())
        avg_score, improved = trainer.train_on_same_board(env2)
        print(f"[OK] Completed. Avg score: {avg_score:.2f}, Improved: {improved}")
        
        # Test 6: Performance test
        print("\n[Test 6] Testing performance...")
        test_score, test_baseline, test_diff = trainer.test_performance(num_tests=3)
        print(f"[OK] Test completed. Avg score: {test_score:.2f}, Diff: {test_diff:+.2f}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR OCCURRED:")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rl_solver()
    sys.exit(0 if success else 1)

