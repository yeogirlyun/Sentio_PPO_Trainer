"""
Distributional Critic for Risk-Aware Value Learning
Implements quantile regression and CVaR optimization for PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QuantileCritic(nn.Module):
    """
    Distributional critic head for quantiles (e.g., 51 uniform τ in [0,1]).
    Outputs: (batch, n_quantiles) for return distribution.
    """
    
    def __init__(self, hidden_size: int, n_quantiles: int = 51, device: Optional[torch.device] = None):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Quantile levels (τ) uniformly distributed in [0,1]
        self.register_buffer('tau', torch.linspace(0, 1, n_quantiles).unsqueeze(0))
        
        # Network layers
        self.head = nn.Linear(hidden_size, n_quantiles)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantile critic.
        
        Args:
            features: Hidden features from policy network (batch, hidden_size)
            
        Returns:
            Quantile values (batch, n_quantiles)
        """
        # Ensure input is on correct device
        features = features.to(self.device)
        
        quantiles = self.head(features)  # (batch, n_quantiles)
        
        # Sort quantiles to ensure monotonicity (optional but helps stability)
        quantiles = torch.sort(quantiles, dim=-1)[0]
        
        return quantiles
    
    def get_value_estimate(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get scalar value estimate (mean of quantiles).
        
        Args:
            features: Hidden features (batch, hidden_size)
            
        Returns:
            Scalar value estimates (batch,)
        """
        quantiles = self.forward(features)
        return quantiles.mean(dim=-1)
    
    def get_cvar(self, features: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            features: Hidden features (batch, hidden_size)
            alpha: Risk level (e.g., 0.05 for 95% CVaR)
            
        Returns:
            CVaR values (batch,)
        """
        quantiles = self.forward(features)
        
        # Get worst α quantiles
        n_worst = max(1, int(self.n_quantiles * alpha))
        worst_quantiles = quantiles[:, :n_worst]
        
        return worst_quantiles.mean(dim=-1)


def quantile_huber_loss(
    pred_quantiles: torch.Tensor, 
    target_quantiles: torch.Tensor, 
    tau: torch.Tensor, 
    kappa: float = 1.0
) -> torch.Tensor:
    """
    Quantile Huber loss for distributional RL.
    
    Args:
        pred_quantiles: Predicted quantiles (batch, n_quantiles)
        target_quantiles: Target quantiles (batch, n_quantiles)
        tau: Quantile levels (1, n_quantiles)
        kappa: Huber threshold
        
    Returns:
        Quantile Huber loss (scalar)
    """
    # Ensure tensors are on same device
    device = pred_quantiles.device
    target_quantiles = target_quantiles.to(device)
    tau = tau.to(device)
    
    # Calculate errors between all pairs of predicted and target quantiles
    # Shape: (batch, n_target_quantiles, n_pred_quantiles)
    err = target_quantiles.unsqueeze(2) - pred_quantiles.unsqueeze(1)
    
    # Indicator function: 1 if error > 0, 0 otherwise
    u = (err > 0).float()
    
    # Huber loss component
    abs_err = torch.abs(err)
    huber = torch.where(
        abs_err <= kappa,
        0.5 * err**2,
        kappa * (abs_err - 0.5 * kappa)
    )
    
    # Quantile loss
    quantile_weight = torch.abs(tau.unsqueeze(1) - (1 - u))
    loss = quantile_weight * huber / kappa
    
    return loss.mean()


class DistributionalValueLoss:
    """
    Enhanced PPO loss with distributional value function and CVaR optimization.
    """
    
    def __init__(
        self, 
        n_quantiles: int = 51, 
        cvar_alpha: float = 0.05, 
        cvar_weight: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.n_quantiles = n_quantiles
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Quantile levels
        self.tau = torch.linspace(0, 1, n_quantiles).unsqueeze(0).to(self.device)
    
    def compute_distributional_loss(
        self, 
        pred_quantiles: torch.Tensor, 
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute distributional value loss with CVaR penalty.
        
        Args:
            pred_quantiles: Predicted quantiles (batch, n_quantiles)
            returns: Target returns (batch,)
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        batch_size = returns.shape[0]
        
        # Create target quantiles from returns
        # Simple approach: use empirical quantiles from returns
        target_quantiles = self._create_target_quantiles(returns)
        
        # Quantile Huber loss
        dist_loss = quantile_huber_loss(pred_quantiles, target_quantiles, self.tau)
        
        # CVaR optimization (Lagrangian penalty)
        cvar_penalty = self._compute_cvar_penalty(pred_quantiles)
        
        # Total loss
        total_loss = dist_loss + self.cvar_weight * cvar_penalty
        
        # Metrics
        metrics = {
            'distributional_loss': dist_loss.item(),
            'cvar_penalty': cvar_penalty.item(),
            'mean_predicted_value': pred_quantiles.mean().item(),
            'cvar_5': self._get_cvar(pred_quantiles, self.cvar_alpha).mean().item()
        }
        
        return total_loss, metrics
    
    def _create_target_quantiles(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Create target quantiles from returns.
        
        Args:
            returns: Target returns (batch,)
            
        Returns:
            Target quantiles (batch, n_quantiles)
        """
        batch_size = returns.shape[0]
        
        # Simple approach: replicate returns across quantiles with some noise
        # In practice, you might use more sophisticated methods
        target_quantiles = returns.unsqueeze(1).repeat(1, self.n_quantiles)
        
        # Add small amount of noise to create distribution
        noise_scale = torch.std(returns) * 0.1
        noise = torch.randn_like(target_quantiles) * noise_scale
        target_quantiles = target_quantiles + noise
        
        # Sort to maintain quantile ordering
        target_quantiles = torch.sort(target_quantiles, dim=-1)[0]
        
        return target_quantiles
    
    def _compute_cvar_penalty(self, quantiles: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR penalty to encourage risk-aware learning.
        
        Args:
            quantiles: Predicted quantiles (batch, n_quantiles)
            
        Returns:
            CVaR penalty (scalar)
        """
        cvar = self._get_cvar(quantiles, self.cvar_alpha)
        
        # Penalty for negative CVaR (encourage positive tail risk management)
        penalty = -cvar.mean()  # Maximize CVaR (minimize negative)
        
        return penalty
    
    def _get_cvar(self, quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Calculate Conditional Value at Risk.
        
        Args:
            quantiles: Quantiles (batch, n_quantiles)
            alpha: Risk level
            
        Returns:
            CVaR values (batch,)
        """
        n_worst = max(1, int(self.n_quantiles * alpha))
        sorted_quantiles = torch.sort(quantiles, dim=-1)[0]
        worst_quantiles = sorted_quantiles[:, :n_worst]
        
        return worst_quantiles.mean(dim=-1)


class EnhancedDistributionalCritic(nn.Module):
    """
    Enhanced critic that combines scalar and distributional value estimation.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: list = [512, 256], 
        n_quantiles: int = 51,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Shared feature extractor
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Scalar value head (for compatibility)
        self.scalar_head = nn.Linear(prev_size, 1)
        
        # Distributional value head
        self.quantile_head = QuantileCritic(prev_size, n_quantiles, device)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, return_distribution: bool = False) -> torch.Tensor:
        """
        Forward pass through critic.
        
        Args:
            x: Input features (batch, input_size)
            return_distribution: Whether to return full distribution
            
        Returns:
            Value estimates or quantiles
        """
        x = x.to(self.device)
        
        # Extract features
        features = self.feature_extractor(x)
        
        if return_distribution:
            # Return full quantile distribution
            return self.quantile_head(features)
        else:
            # Return scalar value estimate
            return self.scalar_head(features).squeeze(-1)
    
    def get_value_and_distribution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both scalar value and quantile distribution.
        
        Args:
            x: Input features (batch, input_size)
            
        Returns:
            Tuple of (scalar_values, quantiles)
        """
        x = x.to(self.device)
        features = self.feature_extractor(x)
        
        scalar_values = self.scalar_head(features).squeeze(-1)
        quantiles = self.quantile_head(features)
        
        return scalar_values, quantiles
