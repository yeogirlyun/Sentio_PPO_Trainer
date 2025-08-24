#!/usr/bin/env python3
"""
Risk-Aware PPO Implementation
Advanced PPO with risk management integration including CVaR optimization, Kelly criterion position sizing,
and comprehensive risk metrics for trading applications.

Features:
- Conditional Value at Risk (CVaR) optimization
- Kelly Criterion for optimal position sizing
- Maximum drawdown constraints
- Sharpe ratio maximization
- Value at Risk (VaR) calculations
- Risk-adjusted reward functions
- Portfolio-level risk management
- Dynamic risk budgeting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics and calculations."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0


@dataclass
class PPORiskConstraints:
    """Risk constraint configuration for trading."""
    max_var_95: float = 0.05        # Maximum 5% VaR
    max_cvar_95: float = 0.08       # Maximum 8% CVaR
    max_drawdown: float = 0.10      # Maximum 10% drawdown
    min_sharpe: float = 1.0         # Minimum Sharpe ratio
    max_volatility: float = 0.25    # Maximum annualized volatility
    max_leverage: float = 2.0       # Maximum leverage
    risk_free_rate: float = 0.02    # Risk-free rate for Sharpe calculation


class RiskCalculator:
    """
    Comprehensive risk calculation utilities for trading strategies.
    """
    
    def __init__(self, lookback_window: int = 252, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize risk calculator.
        
        Args:
            lookback_window: Number of periods for risk calculations
            confidence_levels: Confidence levels for VaR/CVaR calculations
        """
        self.lookback_window = lookback_window
        self.confidence_levels = confidence_levels
        
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level for VaR
            
        Returns:
            VaR value (positive number representing loss)
        """
        if len(returns) == 0:
            return 0.0
        
        return -np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level for CVaR
            
        Returns:
            CVaR value (positive number representing expected loss beyond VaR)
        """
        if len(returns) == 0:
            return 0.0
        
        var_threshold = -self.calculate_var(returns, confidence_level)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
        
        return -np.mean(tail_losses)
    
    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from cumulative returns.
        
        Args:
            cumulative_returns: Array of cumulative returns
            
        Returns:
            Maximum drawdown (positive number)
        """
        if len(cumulative_returns) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return -np.min(drawdown)
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns / np.std(returns) * np.sqrt(252)  # Annualized
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if excess_returns > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return np.inf if excess_returns > 0 else 0.0
        
        return excess_returns / downside_deviation * np.sqrt(252)
    
    def calculate_comprehensive_metrics(self, returns: np.ndarray, 
                                      risk_free_rate: float = 0.02) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        if len(returns) == 0:
            return RiskMetrics()
        
        cumulative_returns = np.cumprod(1 + returns)
        
        metrics = RiskMetrics(
            var_95=self.calculate_var(returns, 0.95),
            var_99=self.calculate_var(returns, 0.99),
            cvar_95=self.calculate_cvar(returns, 0.95),
            cvar_99=self.calculate_cvar(returns, 0.99),
            max_drawdown=self.calculate_max_drawdown(cumulative_returns),
            sharpe_ratio=self.calculate_sharpe_ratio(returns, risk_free_rate),
            sortino_ratio=self.calculate_sortino_ratio(returns, risk_free_rate),
            volatility=np.std(returns) * np.sqrt(252),  # Annualized
            skewness=stats.skew(returns) if len(returns) > 2 else 0.0,
            kurtosis=stats.kurtosis(returns) if len(returns) > 3 else 0.0
        )
        
        # Calculate Calmar ratio (annual return / max drawdown)
        if metrics.max_drawdown > 0:
            annual_return = np.mean(returns) * 252
            metrics.calmar_ratio = annual_return / metrics.max_drawdown
        
        return metrics


class KellyCriterion:
    """
    Kelly Criterion implementation for optimal position sizing.
    """
    
    def __init__(self, lookback_window: int = 100):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            lookback_window: Number of periods to use for probability estimation
        """
        self.lookback_window = lookback_window
        self.win_history = []
        self.loss_history = []
    
    def calculate_kelly_fraction(self, win_prob: float, win_loss_ratio: float, 
                               max_fraction: float = 0.25) -> float:
        """
        Calculate Kelly fraction for position sizing.
        
        Args:
            win_prob: Probability of winning trade
            win_loss_ratio: Ratio of average win to average loss
            max_fraction: Maximum allowed fraction (for safety)
            
        Returns:
            Optimal Kelly fraction (capped at max_fraction)
        """
        if win_prob <= 0 or win_loss_ratio <= 0:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = win_loss_ratio, p = win_prob, q = 1 - win_prob
        kelly_fraction = (win_loss_ratio * win_prob - (1 - win_prob)) / win_loss_ratio
        
        # Cap the fraction for safety
        kelly_fraction = max(0.0, min(kelly_fraction, max_fraction))
        
        return kelly_fraction
    
    def update_trade_outcome(self, pnl: float):
        """
        Update trade history with new outcome.
        
        Args:
            pnl: Profit/loss of the trade
        """
        if pnl > 0:
            self.win_history.append(pnl)
        elif pnl < 0:
            self.loss_history.append(abs(pnl))
        
        # Keep only recent history
        if len(self.win_history) > self.lookback_window:
            self.win_history = self.win_history[-self.lookback_window:]
        if len(self.loss_history) > self.lookback_window:
            self.loss_history = self.loss_history[-self.lookback_window:]
    
    def get_current_kelly_fraction(self, max_fraction: float = 0.25) -> float:
        """
        Get current Kelly fraction based on historical performance.
        
        Args:
            max_fraction: Maximum allowed fraction
            
        Returns:
            Current optimal Kelly fraction
        """
        total_trades = len(self.win_history) + len(self.loss_history)
        
        if total_trades < 10:  # Need minimum history
            return 0.05  # Conservative default
        
        win_prob = len(self.win_history) / total_trades
        
        if len(self.win_history) == 0 or len(self.loss_history) == 0:
            return 0.05  # Conservative default
        
        avg_win = np.mean(self.win_history)
        avg_loss = np.mean(self.loss_history)
        win_loss_ratio = avg_win / avg_loss
        
        return self.calculate_kelly_fraction(win_prob, win_loss_ratio, max_fraction)


class RiskAwarePPO:
    """
    Risk-aware PPO implementation with advanced risk management features.
    
    This class provides:
    - CVaR-based reward adjustment
    - Kelly criterion position sizing
    - Risk constraint enforcement
    - Portfolio-level risk management
    - Dynamic risk budgeting
    """
    
    def __init__(self,
                 risk_constraints: Optional[PPORiskConstraints] = None,
                 cvar_alpha: float = 0.05,
                 kelly_lookback: int = 100,
                 risk_lookback: int = 252,
                 risk_adjustment_factor: float = 1.0,
                 enable_dynamic_sizing: bool = True):
        """
        Initialize Risk-Aware PPO.
        
        Args:
            risk_constraints: Risk constraint configuration
            cvar_alpha: Alpha level for CVaR calculation
            kelly_lookback: Lookback window for Kelly criterion
            risk_lookback: Lookback window for risk calculations
            risk_adjustment_factor: Factor for risk penalty in rewards
            enable_dynamic_sizing: Enable dynamic position sizing
        """
        self.risk_constraints = risk_constraints or PPORiskConstraints()
        self.cvar_alpha = cvar_alpha
        self.risk_adjustment_factor = risk_adjustment_factor
        self.enable_dynamic_sizing = enable_dynamic_sizing
        
        # Initialize components
        self.risk_calculator = RiskCalculator(risk_lookback)
        self.kelly_criterion = KellyCriterion(kelly_lookback)
        
        # Historical data
        self.return_history = []
        self.drawdown_history = []
        self.portfolio_values = []
        self.current_drawdown = 0.0
        
    def compute_cvar_penalty(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR penalty for risk-adjusted rewards.
        
        Args:
            returns: Tensor of returns
            
        Returns:
            CVaR penalty tensor
        """
        if len(returns) == 0:
            return torch.tensor(0.0)
        
        # Convert to numpy for calculation
        returns_np = returns.detach().cpu().numpy()
        
        # Calculate CVaR
        cvar = self.risk_calculator.calculate_cvar(returns_np, 1 - self.cvar_alpha)
        
        # Convert back to tensor
        cvar_penalty = torch.tensor(cvar, dtype=returns.dtype, device=returns.device)
        
        return cvar_penalty
    
    def calculate_optimal_position_size(self, 
                                      signal_strength: float,
                                      current_volatility: float,
                                      available_capital: float) -> float:
        """
        Calculate optimal position size using Kelly criterion and risk constraints.
        
        Args:
            signal_strength: Strength of the trading signal (-1 to 1)
            current_volatility: Current market volatility
            available_capital: Available capital for trading
            
        Returns:
            Optimal position size as fraction of available capital
        """
        if not self.enable_dynamic_sizing:
            return abs(signal_strength) * 0.1  # Fixed 10% sizing
        
        # Get Kelly fraction
        kelly_fraction = self.kelly_criterion.get_current_kelly_fraction()
        
        # Adjust for volatility
        vol_adjustment = min(0.02 / max(current_volatility, 0.001), 2.0)
        adjusted_kelly = kelly_fraction * vol_adjustment
        
        # Apply signal strength
        position_size = abs(signal_strength) * adjusted_kelly
        
        # Apply risk constraints
        max_position = min(
            self.risk_constraints.max_leverage / 10,  # Conservative leverage
            available_capital * 0.2  # Max 20% of capital
        )
        
        return min(position_size, max_position)
    
    def adjust_reward_for_risk(self, 
                              base_reward: float,
                              returns: np.ndarray,
                              current_drawdown: float,
                              portfolio_value: float) -> float:
        """
        Adjust reward based on risk metrics.
        
        Args:
            base_reward: Original reward from environment
            returns: Recent returns history
            current_drawdown: Current portfolio drawdown
            portfolio_value: Current portfolio value
            
        Returns:
            Risk-adjusted reward
        """
        risk_penalty = 0.0
        
        # CVaR penalty
        if len(returns) > 10:
            cvar = self.risk_calculator.calculate_cvar(returns, 1 - self.cvar_alpha)
            if cvar > self.risk_constraints.max_cvar_95:
                risk_penalty += (cvar - self.risk_constraints.max_cvar_95) * 10
        
        # Drawdown penalty
        if current_drawdown > self.risk_constraints.max_drawdown:
            drawdown_penalty = (current_drawdown - self.risk_constraints.max_drawdown) * 20
            risk_penalty += drawdown_penalty
        
        # Volatility penalty
        if len(returns) > 20:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            if volatility > self.risk_constraints.max_volatility:
                vol_penalty = (volatility - self.risk_constraints.max_volatility) * 5
                risk_penalty += vol_penalty
        
        # Sharpe ratio bonus
        if len(returns) > 50:
            sharpe = self.risk_calculator.calculate_sharpe_ratio(returns, 
                                                               self.risk_constraints.risk_free_rate)
            if sharpe > self.risk_constraints.min_sharpe:
                sharpe_bonus = (sharpe - self.risk_constraints.min_sharpe) * 2
                base_reward += sharpe_bonus
        
        # Apply risk adjustment
        adjusted_reward = base_reward - risk_penalty * self.risk_adjustment_factor
        
        return adjusted_reward
    
    def update_portfolio_state(self, 
                             portfolio_value: float,
                             trade_pnl: Optional[float] = None):
        """
        Update portfolio state and risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
            trade_pnl: P&L from last trade (optional)
        """
        # Update portfolio values
        self.portfolio_values.append(portfolio_value)
        
        # Calculate returns if we have previous values
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            return_pct = (portfolio_value - prev_value) / prev_value
            self.return_history.append(return_pct)
        
        # Update Kelly criterion if trade completed
        if trade_pnl is not None:
            self.kelly_criterion.update_trade_outcome(trade_pnl)
        
        # Calculate current drawdown
        if len(self.portfolio_values) > 1:
            peak_value = max(self.portfolio_values)
            self.current_drawdown = (peak_value - portfolio_value) / peak_value
            self.drawdown_history.append(self.current_drawdown)
        
        # Keep history manageable
        max_history = 1000
        if len(self.return_history) > max_history:
            self.return_history = self.return_history[-max_history:]
        if len(self.portfolio_values) > max_history:
            self.portfolio_values = self.portfolio_values[-max_history:]
        if len(self.drawdown_history) > max_history:
            self.drawdown_history = self.drawdown_history[-max_history:]
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        if len(self.return_history) < 10:
            return RiskMetrics()
        
        returns_array = np.array(self.return_history)
        return self.risk_calculator.calculate_comprehensive_metrics(
            returns_array, self.risk_constraints.risk_free_rate
        )
    
    def check_risk_constraints(self) -> Dict[str, bool]:
        """
        Check if current portfolio satisfies risk constraints.
        
        Returns:
            Dictionary of constraint violations
        """
        metrics = self.get_risk_metrics()
        
        violations = {
            'var_95_violation': metrics.var_95 > self.risk_constraints.max_var_95,
            'cvar_95_violation': metrics.cvar_95 > self.risk_constraints.max_cvar_95,
            'drawdown_violation': metrics.max_drawdown > self.risk_constraints.max_drawdown,
            'sharpe_violation': metrics.sharpe_ratio < self.risk_constraints.min_sharpe,
            'volatility_violation': metrics.volatility > self.risk_constraints.max_volatility
        }
        
        return violations
    
    def get_risk_budget_allocation(self, 
                                 strategies: List[str],
                                 strategy_correlations: np.ndarray) -> Dict[str, float]:
        """
        Calculate optimal risk budget allocation across strategies.
        
        Args:
            strategies: List of strategy names
            strategy_correlations: Correlation matrix between strategies
            
        Returns:
            Dictionary of risk budget allocations
        """
        n_strategies = len(strategies)
        
        if n_strategies == 1:
            return {strategies[0]: 1.0}
        
        # Use inverse volatility weighting with correlation adjustment
        try:
            # Calculate inverse correlation matrix
            inv_corr = np.linalg.inv(strategy_correlations)
            
            # Equal risk contribution weights
            weights = np.ones(n_strategies) @ inv_corr
            weights = weights / np.sum(weights)
            
            # Ensure non-negative weights
            weights = np.maximum(weights, 0.0)
            weights = weights / np.sum(weights)
            
            return {strategy: weight for strategy, weight in zip(strategies, weights)}
        
        except np.linalg.LinAlgError:
            # Fallback to equal weighting if correlation matrix is singular
            equal_weight = 1.0 / n_strategies
            return {strategy: equal_weight for strategy in strategies}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = self.get_risk_metrics()
        violations = self.check_risk_constraints()
        
        return {
            'risk_metrics': metrics,
            'constraint_violations': violations,
            'kelly_fraction': self.kelly_criterion.get_current_kelly_fraction(),
            'current_drawdown': self.current_drawdown,
            'total_trades': len(self.kelly_criterion.win_history) + len(self.kelly_criterion.loss_history),
            'win_rate': len(self.kelly_criterion.win_history) / max(
                len(self.kelly_criterion.win_history) + len(self.kelly_criterion.loss_history), 1
            ),
            'portfolio_value': self.portfolio_values[-1] if self.portfolio_values else 0.0
        }
