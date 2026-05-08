"""
PPO Model for Apple Game RL Solver
CPU-optimized Actor-Critic network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# State dimension: board (170) + features (15)
# Features: total_apples(1) + density(1) + avg(1) + max(1) + min(1) + distribution(9) + valid_moves(1) = 15
STATE_DIM = 10 * 17 + 15  # 170 + 15 = 185
MAX_ACTIONS = 500  # Maximum number of possible actions (will be masked)

class ActorCritic(nn.Module):
    """
    PPO Actor-Critic Network.
    CPU-optimized with small network size.
    """
    
    def __init__(self, state_dim=STATE_DIM, hidden_dim=128):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state vector
            hidden_dim: Hidden layer dimension (small for CPU)
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, MAX_ACTIONS)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            action_logits: Action logits (batch_size, MAX_ACTIONS)
            value: State value (batch_size, 1)
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state, valid_actions, deterministic=False):
        """
        Get action from policy.
        
        Args:
            state: State tensor of shape (state_dim,)
            valid_actions: List of valid actions (tuples of (r1, c1, r2, c2))
            deterministic: If True, return best action; else sample from distribution
        
        Returns:
            action: Selected action tuple
            action_idx: Index of selected action in valid_actions list
            log_prob: Log probability of selected action
            value: State value
        """
        self.eval()
        with torch.no_grad():
            if not valid_actions:
                # No valid actions
                return None, 0, torch.tensor(0.0), torch.tensor(0.0)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, value = self.forward(state_tensor)
            
            # Limit to number of valid actions
            num_valid = min(len(valid_actions), MAX_ACTIONS)
            valid_logits = action_logits[0, :num_valid]
            
            # Apply temperature scaling to prevent extreme values
            valid_logits = valid_logits / 1.0  # Temperature = 1.0
            
            # Clamp to prevent inf/nan
            valid_logits = torch.clamp(valid_logits, min=-50, max=50)
            
            if deterministic:
                action_idx = valid_logits.argmax().item()
            else:
                # Sample from distribution
                probs = F.softmax(valid_logits, dim=0)
                # Check for valid probabilities
                if torch.any(torch.isnan(probs)) or torch.any(probs < 0):
                    # Fallback to uniform distribution
                    probs = torch.ones_like(probs) / len(probs)
                action_idx = torch.multinomial(probs, 1).item()
            
            # Get log probability
            log_prob = F.log_softmax(valid_logits, dim=0)[action_idx]
            
            # Get actual action
            actual_action = valid_actions[action_idx]
            
            return actual_action, action_idx, log_prob, value.squeeze()
    
    def evaluate_actions(self, states, actions, valid_actions_list):
        """
        Evaluate actions for PPO update.
        
        Args:
            states: State tensor (batch_size, state_dim)
            actions: Action indices (batch_size,)
            valid_actions_list: List of valid actions for each state
        
        Returns:
            action_log_probs: Log probabilities of actions
            values: State values
            entropy: Entropy of policy
        """
        action_logits, values = self.forward(states)
        
        # Create masks for each state
        batch_size = states.size(0)
        masked_logits = []
        
        for i in range(batch_size):
            mask = torch.zeros(MAX_ACTIONS, dtype=torch.bool)
            if i < len(valid_actions_list) and len(valid_actions_list[i]) > 0:
                valid_indices = list(range(min(len(valid_actions_list[i]), MAX_ACTIONS)))
                mask[valid_indices] = True
            else:
                mask[0] = True  # At least one valid action
            
            masked_logit = action_logits[i].clone()
            masked_logit[~mask] = float('-inf')
            masked_logits.append(masked_logit)
        
        masked_logits = torch.stack(masked_logits)
        probs = F.softmax(masked_logits, dim=1)
        log_probs = F.log_softmax(masked_logits, dim=1)
        
        # Get log probs for selected actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate entropy
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        return action_log_probs, values.squeeze(), entropy

