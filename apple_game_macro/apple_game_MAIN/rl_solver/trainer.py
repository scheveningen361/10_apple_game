"""
PPO Trainer for Apple Game RL Solver
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

from rl_solver.environment import AppleGameEnv
from rl_solver.model import ActorCritic
from rl_solver.utils import generate_random_board, save_model, load_model

class PPOTrainer:
    """
    PPO Trainer with same-board repetition and checkpoint saving.
    """
    
    def __init__(
        self,
        state_dim=185,  # ROWS*COLS (170) + features (15) = 185
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        save_interval=100,
        test_interval=50,
        same_board_episodes=20,
        improvement_threshold=0.1
    ):
        """
        Initialize PPO Trainer.
        
        Args:
            state_dim: State dimension
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clip parameter
            k_epochs: Number of update epochs
            save_interval: Episodes between model saves
            test_interval: Episodes between performance tests
            same_board_episodes: Number of episodes on same board before switching
            improvement_threshold: Minimum improvement to continue on same board
        """
        self.device = torch.device("cpu")
        self.model = ActorCritic(state_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.save_interval = save_interval
        self.test_interval = test_interval
        self.same_board_episodes = same_board_episodes
        self.improvement_threshold = improvement_threshold
        
        self.episode = 0
        self.best_score = float('-inf')
        self.recent_scores = deque(maxlen=same_board_episodes)
        
        # Create saved_models directory if it doesn't exist
        os.makedirs("rl_solver/saved_models", exist_ok=True)
    
    def compute_returns(self, rewards, dones, next_value=0):
        """Compute discounted returns."""
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = reward
            else:
                R = reward + self.gamma * R
            returns.insert(0, R)
        return returns
    
    def update(self, states, actions, old_log_probs, returns, advantages, valid_actions_list):
        """PPO update step."""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Convert advantages to numpy array and normalize
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize returns (already converted to tensor)
        returns_mean = returns.mean()
        returns_std = returns.std()
        if returns_std > 1e-8:
            returns = (returns - returns_mean) / returns_std
        
        for _ in range(self.k_epochs):
            # Evaluate actions
            new_log_probs, values, entropy = self.model.evaluate_actions(
                states, actions, valid_actions_list
            )
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
    
    def train_episode(self, env, callback=None):
        """
        Train one episode.
        
        Args:
            env: Game environment
            callback: Optional callback function for progress updates
        
        Returns:
            total_reward: Total reward for episode
            score: Final score
            baseline_score: Baseline score
        """
        state = env.reset()
        states = []
        actions = []
        action_indices = []
        old_log_probs = []
        rewards = []
        dones = []
        valid_actions_list = []
        
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            valid_actions_list.append(valid_actions)
            
            # Get action from policy
            action, action_idx, log_prob, value = self.model.get_action(state, valid_actions)
            
            if action is None:
                # No valid actions
                break
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            action_indices.append(action_idx)
            old_log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            
            if done:
                break
            
            state = next_state
        
        # Compute returns and advantages
        final_value = 0 if dones[-1] else self.model.get_action(state, env.get_valid_actions())[2].item()
        returns = self.compute_returns(rewards, dones, final_value)
        
        # Compute advantages (simple: returns - values, but we don't have values stored)
        # For simplicity, use returns as advantages (REINFORCE-style)
        advantages = returns
        
        # Update policy
        if len(states) > 0:
            self.update(states, action_indices, old_log_probs, returns, advantages, valid_actions_list)
        
        total_reward = sum(rewards)
        score = env.get_score()
        baseline_score = env.get_baseline_score()
        
        self.episode += 1
        
        if callback:
            callback(self.episode, score, baseline_score, total_reward)
        
        return total_reward, score, baseline_score
    
    def train_on_same_board(self, env, callback=None):
        """
        Train multiple episodes on the same board.
        
        Args:
            env: Game environment
            callback: Optional callback function
        
        Returns:
            avg_score: Average score on this board
            improved: Whether score improved
        """
        scores = []
        initial_score = None
        
        for ep in range(self.same_board_episodes):
            _, score, baseline = self.train_episode(env, callback)
            scores.append(score)
            self.recent_scores.append(score)
            
            if initial_score is None:
                initial_score = score
            
            # Check for improvement
            if ep >= 5:  # After some episodes
                recent_avg = np.mean(list(self.recent_scores)[-5:])
                if recent_avg > initial_score + self.improvement_threshold:
                    # Improved, continue
                    pass
                elif ep >= 10:  # Give it more time
                    # No improvement after many episodes, might switch board
                    pass
        
        avg_score = np.mean(scores)
        improved = avg_score > initial_score + self.improvement_threshold
        
        return avg_score, improved
    
    def test_performance(self, num_tests=10):
        """
        Test model performance on random boards.
        
        Args:
            num_tests: Number of test games
        
        Returns:
            avg_score: Average score
            avg_baseline: Average baseline score
            avg_diff: Average difference
        """
        scores = []
        baselines = []
        
        for _ in range(num_tests):
            board = generate_random_board()
            env = AppleGameEnv(board)
            state = env.reset()
            
            while not env.done:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                action, _, _, _ = self.model.get_action(state, valid_actions, deterministic=True)
                state, _, done, _ = env.step(action)
                if done:
                    break
            
            scores.append(env.get_score())
            baselines.append(env.get_baseline_score())
        
        avg_score = np.mean(scores)
        avg_baseline = np.mean(baselines)
        avg_diff = avg_score - avg_baseline
        
        return avg_score, avg_baseline, avg_diff
    
    def save_checkpoint(self, path=None):
        """Save model checkpoint."""
        if path is None:
            path = f"rl_solver/saved_models/checkpoint_ep{self.episode}.pt"
        save_model(self.model, path, self.episode, self.best_score)
        return path
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        episode, score = load_model(self.model, path)
        self.episode = episode
        self.best_score = score
        return episode, score

