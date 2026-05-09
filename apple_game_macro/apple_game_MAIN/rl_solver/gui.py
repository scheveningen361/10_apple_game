"""
GUI for RL Training and Testing
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from rl_solver.trainer import PPOTrainer
from rl_solver.environment import AppleGameEnv
from rl_solver.utils import generate_random_board, load_model

class RLTrainingGUI:
    """GUI for RL training and testing."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RL Apple Game Solver - Training & Testing")
        self.root.geometry("900x700")
        
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.start_time = None
        self.training_duration = None  # in seconds
        
        # Training statistics
        self.episode_scores = []
        self.episode_baselines = []
        self.episode_diffs = []
        self.test_scores = []
        self.test_baselines = []
        self.test_diffs = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Training settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Training Settings", padding="10")
        settings_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Training duration
        ttk.Label(settings_frame, text="Training Duration (hours):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.duration_var = tk.StringVar(value="1.0")
        duration_entry = ttk.Entry(settings_frame, textvariable=self.duration_var, width=10)
        duration_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.grid(row=0, column=2, columnspan=2, padx=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress", padding="10")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Status labels
        self.status_label = ttk.Label(progress_frame, text="Ready to train")
        self.status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        self.episode_label = ttk.Label(progress_frame, text="Episode: 0")
        self.episode_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        self.score_label = ttk.Label(progress_frame, text="Score: - / Baseline: -")
        self.score_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=2)
        
        self.time_label = ttk.Label(progress_frame, text="Elapsed: 00:00:00 / Remaining: --:--:--")
        self.time_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=80)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        # Test frame
        test_frame = ttk.LabelFrame(main_frame, text="Model Testing", padding="10")
        test_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(test_frame, text="Test Current Model", command=self.test_model).grid(row=0, column=0, padx=5)
        ttk.Button(test_frame, text="Load Model", command=self.load_model).grid(row=0, column=1, padx=5)
        
        self.test_result_label = ttk.Label(test_frame, text="No test results yet")
        self.test_result_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        test_frame.columnconfigure(0, weight=1)
    
    def update_progress(self, episode, score, baseline, reward):
        """Update progress display."""
        self.episode_scores.append(score)
        self.episode_baselines.append(baseline)
        self.episode_diffs.append(score - baseline)
        
        self.episode_label.config(text=f"Episode: {episode}")
        self.score_label.config(text=f"Score: {score} / Baseline: {baseline} (Diff: {score - baseline:+d})")
        
        # Update statistics
        if len(self.episode_scores) > 0:
            avg_score = np.mean(self.episode_scores[-50:])  # Last 50 episodes
            avg_baseline = np.mean(self.episode_baselines[-50:])
            avg_diff = np.mean(self.episode_diffs[-50:])
            
            stats_text = f"Recent 50 Episodes:\n"
            stats_text += f"  Average Score: {avg_score:.2f}\n"
            stats_text += f"  Average Baseline: {avg_baseline:.2f}\n"
            stats_text += f"  Average Difference: {avg_diff:+.2f}\n"
            stats_text += f"\nTotal Episodes: {episode}\n"
            stats_text += f"Best Score: {max(self.episode_scores) if self.episode_scores else 0}\n"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
    
    def update_time(self):
        """Update elapsed and remaining time."""
        if self.is_training and self.start_time:
            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            if self.training_duration:
                remaining = max(0, self.training_duration - elapsed)
                remaining_str = str(timedelta(seconds=int(remaining)))
                progress = min(100, (elapsed / self.training_duration) * 100)
                self.progress_var.set(progress)
            else:
                remaining_str = "--:--:--"
            
            self.time_label.config(text=f"Elapsed: {elapsed_str} / Remaining: {remaining_str}")
            
            if self.is_training:
                self.root.after(1000, self.update_time)
    
    def start_training(self):
        """Start training thread."""
        try:
            duration_hours = float(self.duration_var.get())
            if duration_hours <= 0:
                messagebox.showerror("Error", "Training duration must be positive")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid training duration")
            return
        
        self.training_duration = duration_hours * 3600  # Convert to seconds
        self.is_training = True
        self.start_time = time.time()
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Training in progress...")
        
        # Reset statistics
        self.episode_scores = []
        self.episode_baselines = []
        self.episode_diffs = []
        
        # Start training thread
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
        
        # Start time update
        self.update_time()
    
    def stop_training(self):
        """Stop training."""
        self.is_training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Training stopped")
    
    def training_loop(self):
        """Main training loop."""
        # Initialize trainer
        self.trainer = PPOTrainer(
            state_dim=185,
            hidden_dim=128,
            lr=3e-4,
            save_interval=100,
            test_interval=50,
            same_board_episodes=20
        )
        
        current_board = generate_random_board()
        env = AppleGameEnv(current_board)
        last_test_time = 0
        last_save_time = 0
        
        while self.is_training:
            # Check if training time is up
            if self.training_duration and (time.time() - self.start_time) >= self.training_duration:
                self.is_training = False
                self.root.after(0, lambda: self.status_label.config(text="Training completed"))
                break
            
            # Train on same board
            avg_score, improved = self.trainer.train_on_same_board(
                env,
                callback=lambda ep, score, baseline, reward: self.root.after(
                    0, lambda: self.update_progress(ep, score, baseline, reward)
                )
            )
            
            # Performance test
            current_time = time.time()
            if current_time - last_test_time >= 300:  # Every 5 minutes
                test_score, test_baseline, test_diff = self.trainer.test_performance(num_tests=10)
                self.test_scores.append(test_score)
                self.test_baselines.append(test_baseline)
                self.test_diffs.append(test_diff)
                last_test_time = current_time
                
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Test: Avg Score={test_score:.1f}, Diff={test_diff:+.1f}"
                ))
            
            # Save checkpoint
            if self.trainer.episode - last_save_time >= self.trainer.save_interval:
                checkpoint_path = self.trainer.save_checkpoint()
                last_save_time = self.trainer.episode
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Checkpoint saved: {checkpoint_path}"
                ))
            
            # Switch to new board if no improvement
            if not improved or random.random() < 0.1:  # 10% chance to switch anyway
                current_board = generate_random_board()
                env = AppleGameEnv(current_board)
        
        # Final save
        if self.trainer:
            final_path = self.trainer.save_checkpoint("rl_solver/saved_models/final_model.pt")
            self.root.after(0, lambda: self.status_label.config(text=f"Training finished. Model saved: {final_path}"))
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
    
    def test_model(self):
        """Test current model."""
        if not self.trainer:
            messagebox.showwarning("Warning", "No model loaded. Start training first.")
            return
        
        self.test_result_label.config(text="Testing...")
        self.root.update()
        
        test_score, test_baseline, test_diff = self.trainer.test_performance(num_tests=20)
        
        result_text = f"Test Results (20 games):\n"
        result_text += f"  Average Score: {test_score:.2f}\n"
        result_text += f"  Average Baseline: {test_baseline:.2f}\n"
        result_text += f"  Average Difference: {test_diff:+.2f}"
        
        self.test_result_label.config(text=result_text)
    
    def load_model(self):
        """Load model from file."""
        file_path = filedialog.askopenfilename(
            initialdir="rl_solver/saved_models",
            title="Select model file",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if not self.trainer:
                    self.trainer = PPOTrainer(state_dim=185, hidden_dim=128)
                
                episode, score = self.trainer.load_checkpoint(file_path)
                messagebox.showinfo("Success", f"Model loaded from episode {episode}, score: {score}")
                self.status_label.config(text=f"Model loaded: Episode {episode}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

def main():
    """Main entry point."""
    root = tk.Tk()
    app = RLTrainingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

