#!/usr/bin/env python3
"""
Advanced PPO Loss Implementation
Enhanced PPO loss with adaptive KL divergence, trust region constraints, and multi-step returns.

Features:
- Adaptive KL divergence penalty for trust region enforcement
- Multi-step temporal difference learning
- Clipped surrogate objective (PPO-2 style)
- Dynamic coefficient adjustment based on KL divergence
- Comprehensive loss component tracking
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedPPOLoss:
    """
    Advanced PPO loss with adaptive KL, trust region, and multi-step returns.
    
    This implementation includes:
    - Adaptive KL divergence penalty that adjusts based on policy updates
    - Trust region constraints to prevent large policy changes
    - Multi-step temporal difference returns for better value estimation
    - Comprehensive loss component tracking for monitoring
    """
    
    def __init__(self, 
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 target_kl: float = 0.01,
                 n_steps: int = 5,
                 kl_coef_init: float = 1.0,
                 kl_adapt_factor: float = 1.5,
                 max_grad_norm: float = 0.5):
        """
        Initialize the Advanced PPO Loss.
        
        Args:
            clip_epsilon: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            target_kl: Target KL divergence for adaptive penalty
            n_steps: Number of steps for multi-step returns
            kl_coef_init: Initial KL coefficient
            kl_adapt_factor: Factor for KL coefficient adaptation
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.n_steps = n_steps
        self.kl_coef = kl_coef_init
        self.kl_adapt_factor = kl_adapt_factor
        self.max_grad_norm = max_grad_norm
        
        # Tracking variables
        self.loss_history = []
        self.kl_history = []
        
    def compute_gae_returns(self, 
                           rewards: torch.Tensor, 
                           values: torch.Tensor, 
                           dones: torch.Tensor,
                           gamma: float = 0.99,
                           gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) returns and advantages.
        
        Args:
            rewards: Reward tensor [batch_size, seq_len]
            values: Value function estimates [batch_size, seq_len]
            dones: Done flags [batch_size, seq_len]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (returns, advantages)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
                next_done = 1
            else:
                next_value = values[:, t + 1]
                next_done = dones[:, t + 1]
            
            delta = rewards[:, t] + gamma * next_value * (1 - next_done) - values[:, t]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]
            
        return returns, advantages
    
    def compute_multi_step_returns(self, 
                                  rewards: torch.Tensor, 
                                  values: torch.Tensor,
                                  dones: torch.Tensor,
                                  gamma: float = 0.99) -> torch.Tensor:
        """
        Compute multi-step returns for improved value estimation.
        
        Args:
            rewards: Reward tensor
            values: Value estimates
            dones: Done flags
            gamma: Discount factor
            
        Returns:
            Multi-step returns tensor
        """
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        
        for t in range(seq_len):
            return_val = 0
            for k in range(min(self.n_steps, seq_len - t)):
                if t + k < seq_len:
                    return_val += (gamma ** k) * rewards[:, t + k]
                    if dones[:, t + k].any():
                        break
            
            # Add discounted value estimate at n-step horizon
            if t + self.n_steps < seq_len:
                return_val += (gamma ** self.n_steps) * values[:, t + self.n_steps]
            
            returns[:, t] = return_val
            
        return returns
    
    def compute_loss(self, 
                    old_log_probs: torch.Tensor,
                    new_log_probs: torch.Tensor,
                    advantages: torch.Tensor,
                    returns: torch.Tensor,
                    values: torch.Tensor,
                    actions: torch.Tensor,
                    entropy: torch.Tensor,
                    rewards: Optional[torch.Tensor] = None,
                    dones: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the advanced PPO loss with all enhancements.
        
        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy
            advantages: Advantage estimates
            returns: Return estimates
            values: Value function estimates
            actions: Actions taken
            entropy: Policy entropy
            rewards: Raw rewards (optional, for multi-step returns)
            dones: Done flags (optional, for multi-step returns)
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective (PPO-2 style)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss with multi-step returns if available
        if rewards is not None and dones is not None:
            multi_step_returns = self.compute_multi_step_returns(rewards, values, dones)
            value_loss = F.mse_loss(values.squeeze(), multi_step_returns)
        else:
            value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy bonus for exploration
        entropy_loss = -entropy.mean()
        
        # Adaptive KL penalty (trust region constraint)
        kl_div = (old_log_probs - new_log_probs).mean()
        
        # Adapt KL coefficient based on divergence
        if kl_div > self.kl_adapt_factor * self.target_kl:
            self.kl_coef *= self.kl_adapt_factor
            logger.debug(f"Increased KL coefficient to {self.kl_coef:.4f}")
        elif kl_div < self.target_kl / self.kl_adapt_factor:
            self.kl_coef /= self.kl_adapt_factor
            logger.debug(f"Decreased KL coefficient to {self.kl_coef:.4f}")
        
        # KL penalty loss
        kl_loss = self.kl_coef * kl_div
        
        # Total loss combination
        total_loss = (policy_loss + 
                     self.value_coef * value_loss + 
                     self.entropy_coef * entropy_loss + 
                     kl_loss)
        
        # Collect metrics for monitoring
        metrics = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "kl_divergence": kl_div.item(),
            "kl_loss": kl_loss.item(),
            "kl_coefficient": self.kl_coef,
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "clip_fraction": ((ratio > 1 + self.clip_epsilon) | (ratio < 1 - self.clip_epsilon)).float().mean().item()
        }
        
        # Store history for analysis
        self.loss_history.append(total_loss.item())
        self.kl_history.append(kl_div.item())
        
        return total_loss, metrics
    
    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics for analysis."""
        if not self.loss_history:
            return {}
            
        return {
            "loss_mean": np.mean(self.loss_history[-100:]),  # Last 100 steps
            "loss_std": np.std(self.loss_history[-100:]),
            "kl_mean": np.mean(self.kl_history[-100:]),
            "kl_std": np.std(self.kl_history[-100:]),
            "current_kl_coef": self.kl_coef
        }
    
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.loss_history.clear()
        self.kl_history.clear()


class TrustRegionConstraint:
    """
    Additional trust region constraint for more sophisticated policy updates.
    """
    
    def __init__(self, max_kl: float = 0.01, backtrack_coef: float = 0.8, max_backtracks: int = 10):
        self.max_kl = max_kl
        self.backtrack_coef = backtrack_coef
        self.max_backtracks = max_backtracks
    
    def line_search(self, 
                   model: torch.nn.Module,
                   old_params: Dict[str, torch.Tensor],
                   gradient: Dict[str, torch.Tensor],
                   old_log_probs: torch.Tensor,
                   states: torch.Tensor,
                   actions: torch.Tensor) -> bool:
        """
        Perform line search to ensure trust region constraint.
        
        Returns:
            True if acceptable step found, False otherwise
        """
        step_size = 1.0
        
        for _ in range(self.max_backtracks):
            # Apply step
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in gradient:
                        param.data = old_params[name] - step_size * gradient[name]
            
            # Check KL constraint
            with torch.no_grad():
                new_log_probs = model.get_log_probs(states, actions)
                kl_div = (old_log_probs - new_log_probs).mean()
            
            if kl_div <= self.max_kl:
                return True
            
            step_size *= self.backtrack_coef
        
        # Restore original parameters if no acceptable step found
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in old_params:
                    param.data = old_params[name]
        
        return False
