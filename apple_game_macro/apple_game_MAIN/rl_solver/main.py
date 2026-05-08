"""
Main entry point for RL Solver GUI
Can be run from rl_solver directory or project root.
"""

import sys
import os

# Add parent directory to path if running from rl_solver directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from rl_solver.gui import main

if __name__ == "__main__":
    main()

