"""
Regime Gating for Market-Aware PPO Training
Implements HMM-based market regime detection for adaptive policy switching.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Graceful import handling for optional dependencies
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    HMM_AVAILABLE = True
except ImportError:
    logger.warning("statsmodels not available. Using fallback regime detection.")
    HMM_AVAILABLE = False


@dataclass
class RegimeMetrics:
    """Metrics for a specific market regime."""
    trades: int = 0
    total_pnl: float = 0.0
    sharpe_values: List[float] = None
    win_rate: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    
    def __post_init__(self):
        if self.sharpe_values is None:
            self.sharpe_values = []
    
    @property
    def avg_sharpe(self) -> float:
        """Calculate average Sharpe ratio."""
        return np.mean(self.sharpe_values) if self.sharpe_values else 0.0


class RegimeGater:
    """
    Regime classifier (HMM-based) for gating policies/masks.
    
    Detects market regimes and provides adaptive policy switching:
    - Low volatility (ranging/choppy markets)
    - Medium volatility (mixed conditions)  
    - High volatility (trending/news-driven)
    """
    
    def __init__(
        self, 
        n_regimes: int = 3, 
        vol_thresholds: List[float] = [0.01, 0.03],
        lookback_window: int = 100,
        min_samples_fit: int = 200
    ):
        self.n_regimes = n_regimes
        self.vol_thresholds = vol_thresholds
        self.lookback_window = lookback_window
        self.min_samples_fit = min_samples_fit
        
        # HMM model (if available)
        self.hmm_model = None
        self.is_fitted = False
        
        # Regime history tracking
        self.regime_history: Dict[int, RegimeMetrics] = {
            i: RegimeMetrics() for i in range(n_regimes)
        }
        
        # Current regime state
        self.current_regime = 0
        self.regime_confidence = 0.0
        
        # Feature history for regime detection
        self.feature_history = []
        
    def fit_hmm(self, returns_series: pd.Series) -> bool:
        """
        Fit HMM on historical returns.
        
        Args:
            returns_series: Historical returns for regime fitting
            
        Returns:
            True if fitting successful, False otherwise
        """
        if not HMM_AVAILABLE:
            logger.info("HMM not available, using fallback regime detection")
            return False
            
        if len(returns_series) < self.min_samples_fit:
            logger.warning(f"Insufficient data for HMM fitting: {len(returns_series)} < {self.min_samples_fit}")
            return False
        
        try:
            # Prepare data - remove NaN values
            clean_returns = returns_series.dropna()
            
            if len(clean_returns) < self.min_samples_fit:
                logger.warning("Insufficient clean data after removing NaN values")
                return False
            
            # Fit Markov Regression model
            self.hmm_model = MarkovRegression(
                clean_returns, 
                k_regimes=self.n_regimes, 
                trend='c',
                switching_variance=True
            )
            
            # Fit with error handling
            self.hmm_model = self.hmm_model.fit(maxiter=100, disp=False)
            self.is_fitted = True
            
            logger.info(f"HMM fitted successfully with {self.n_regimes} regimes")
            return True
            
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            self.hmm_model = None
            self.is_fitted = False
            return False
    
    def detect_regime(self, recent_returns: np.ndarray, market_features: Optional[Dict[str, float]] = None) -> int:
        """
        Detect current market regime.
        
        Args:
            recent_returns: Recent return observations
            market_features: Additional market features (volume, spread, etc.)
            
        Returns:
            Regime index (0 to n_regimes-1)
        """
        if len(recent_returns) == 0:
            return self.current_regime
        
        # Store features for history
        self._update_feature_history(recent_returns, market_features)
        
        # Try HMM-based detection first
        if self.is_fitted and self.hmm_model is not None:
            regime = self._hmm_regime_detection(recent_returns)
            if regime is not None:
                self.current_regime = regime
                return regime
        
        # Fallback to volatility-based detection
        regime = self._volatility_based_detection(recent_returns, market_features)
        self.current_regime = regime
        return regime
    
    def _hmm_regime_detection(self, recent_returns: np.ndarray) -> Optional[int]:
        """
        HMM-based regime detection.
        
        Args:
            recent_returns: Recent return observations
            
        Returns:
            Regime index or None if detection fails
        """
        try:
            # Use the last return for prediction
            last_return = recent_returns[-1] if len(recent_returns) > 0 else 0.0
            
            # Get regime probabilities
            regime_probs = self.hmm_model.predict(pd.Series([last_return]))
            
            if len(regime_probs) > 0:
                # Get most likely regime
                regime = np.argmax(regime_probs[-1])
                self.regime_confidence = np.max(regime_probs[-1])
                return int(regime)
                
        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            
        return None
    
    def _volatility_based_detection(self, recent_returns: np.ndarray, market_features: Optional[Dict[str, float]] = None) -> int:
        """
        Fallback volatility-based regime detection.
        
        Args:
            recent_returns: Recent return observations
            market_features: Additional market features
            
        Returns:
            Regime index
        """
        # Calculate volatility
        vol = np.std(recent_returns) if len(recent_returns) > 1 else 0.0
        
        # Incorporate additional features if available
        if market_features:
            # Adjust volatility based on volume, spread, etc.
            volume_factor = market_features.get('volume_ratio', 1.0)
            spread_factor = market_features.get('spread_ratio', 1.0)
            
            # Higher volume and spread typically indicate higher volatility regime
            vol = vol * (0.7 + 0.3 * volume_factor) * (0.8 + 0.2 * spread_factor)
        
        # Classify based on thresholds
        if vol < self.vol_thresholds[0]:
            regime = 0  # Low volatility (ranging/choppy)
            self.regime_confidence = 1.0 - (vol / self.vol_thresholds[0])
        elif vol < self.vol_thresholds[1]:
            regime = 1  # Medium volatility (mixed)
            self.regime_confidence = 0.7
        else:
            regime = 2  # High volatility (trending/news-driven)
            self.regime_confidence = min(1.0, vol / self.vol_thresholds[1] - 1.0)
        
        return regime
    
    def _update_feature_history(self, returns: np.ndarray, market_features: Optional[Dict[str, float]] = None):
        """Update feature history for regime detection."""
        feature_dict = {
            'timestamp': pd.Timestamp.now(),
            'returns': returns.copy() if len(returns) > 0 else np.array([0.0]),
            'volatility': np.std(returns) if len(returns) > 1 else 0.0,
            'mean_return': np.mean(returns) if len(returns) > 0 else 0.0
        }
        
        if market_features:
            feature_dict.update(market_features)
        
        self.feature_history.append(feature_dict)
        
        # Keep only recent history
        if len(self.feature_history) > self.lookback_window:
            self.feature_history = self.feature_history[-self.lookback_window:]
    
    def log_per_regime(self, regime: int, pnl: float, trade_count: int = 1, returns: Optional[np.ndarray] = None):
        """
        Log performance metrics per regime.
        
        Args:
            regime: Regime index
            pnl: Profit/loss for the period
            trade_count: Number of trades
            returns: Return series for Sharpe calculation
        """
        if regime not in self.regime_history:
            logger.warning(f"Invalid regime index: {regime}")
            return
        
        metrics = self.regime_history[regime]
        
        # Update basic metrics
        metrics.trades += trade_count
        metrics.total_pnl += pnl
        
        # Calculate Sharpe ratio if returns provided
        if returns is not None and len(returns) > 0:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret > 0:
                sharpe = mean_ret / std_ret * np.sqrt(252 * 390)  # Annualized
                metrics.sharpe_values.append(sharpe)
        
        # Update derived metrics
        if metrics.trades > 0:
            metrics.avg_return = metrics.total_pnl / metrics.trades
            
        # Calculate win rate (simplified)
        if pnl > 0:
            metrics.win_rate = (metrics.win_rate * (metrics.trades - trade_count) + trade_count) / metrics.trades
        else:
            metrics.win_rate = (metrics.win_rate * (metrics.trades - trade_count)) / metrics.trades
    
    def get_regime_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Get performance metrics for all regimes.
        
        Returns:
            Dictionary mapping regime index to metrics
        """
        result = {}
        
        for regime_idx, metrics in self.regime_history.items():
            result[regime_idx] = {
                'total_trades': metrics.trades,
                'total_pnl': metrics.total_pnl,
                'avg_sharpe': metrics.avg_sharpe,
                'win_rate': metrics.win_rate,
                'avg_return': metrics.avg_return,
                'volatility': metrics.volatility
            }
        
        return result
    
    def get_regime_expectancy(self, regime: int) -> float:
        """
        Get expected return for a specific regime.
        
        Args:
            regime: Regime index
            
        Returns:
            Expected return (positive indicates profitable regime)
        """
        if regime not in self.regime_history:
            return 0.0
        
        metrics = self.regime_history[regime]
        
        if metrics.trades == 0:
            return 0.0
        
        # Simple expectancy calculation
        return metrics.avg_return
    
    def should_trade_in_regime(self, regime: int, min_expectancy: float = 0.0) -> bool:
        """
        Determine if trading should be allowed in a specific regime.
        
        Args:
            regime: Regime index
            min_expectancy: Minimum expected return threshold
            
        Returns:
            True if trading is recommended
        """
        expectancy = self.get_regime_expectancy(regime)
        return expectancy >= min_expectancy
    
    def get_regime_info(self) -> Dict[str, Any]:
        """
        Get current regime information.
        
        Returns:
            Dictionary with current regime details
        """
        return {
            'current_regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'is_hmm_fitted': self.is_fitted,
            'regime_names': {
                0: 'Low Volatility (Ranging)',
                1: 'Medium Volatility (Mixed)',
                2: 'High Volatility (Trending)'
            },
            'current_regime_name': {
                0: 'Low Volatility (Ranging)',
                1: 'Medium Volatility (Mixed)', 
                2: 'High Volatility (Trending)'
            }.get(self.current_regime, 'Unknown')
        }


class RegimeAwareMasker:
    """
    Enhanced action masker that considers market regimes.
    """
    
    def __init__(self, regime_gater: RegimeGater, base_masker=None):
        self.regime_gater = regime_gater
        self.base_masker = base_masker
        
        # Regime-specific action preferences
        self.regime_action_preferences = {
            0: {'hold_bias': 0.8, 'aggressive_penalty': 0.5},  # Low vol: prefer holding
            1: {'hold_bias': 0.5, 'aggressive_penalty': 0.2},  # Med vol: balanced
            2: {'hold_bias': 0.2, 'aggressive_penalty': 0.0}   # High vol: allow aggressive
        }
    
    def get_action_mask(
        self, 
        observation: np.ndarray, 
        recent_returns: np.ndarray,
        market_features: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Get regime-aware action mask.
        
        Args:
            observation: Current market observation
            recent_returns: Recent return history
            market_features: Additional market features
            
        Returns:
            Action mask (1 = allowed, 0 = masked)
        """
        # Detect current regime
        current_regime = self.regime_gater.detect_regime(recent_returns, market_features)
        
        # Start with base mask if available
        if self.base_masker is not None:
            base_mask = self.base_masker.get_action_mask(observation)
        else:
            # Default: all actions allowed (assuming 3 actions: sell, hold, buy)
            base_mask = np.ones(3, dtype=np.float32)
        
        # Apply regime-specific modifications
        regime_prefs = self.regime_action_preferences.get(current_regime, {})
        
        # Modify mask based on regime preferences
        modified_mask = base_mask.copy()
        
        # Example: In low volatility regime, reduce aggressive actions
        if current_regime == 0:  # Low volatility
            # Reduce probability of extreme actions (buy/sell), favor hold
            if len(modified_mask) >= 3:
                modified_mask[0] *= (1 - regime_prefs.get('aggressive_penalty', 0))  # Sell
                modified_mask[2] *= (1 - regime_prefs.get('aggressive_penalty', 0))  # Buy
                modified_mask[1] *= (1 + regime_prefs.get('hold_bias', 0))          # Hold
        
        # Normalize to ensure valid probabilities
        if np.sum(modified_mask) > 0:
            modified_mask = modified_mask / np.sum(modified_mask)
        else:
            modified_mask = base_mask  # Fallback to base mask
        
        return modified_mask
    
    def get_regime_trading_signal(self, regime: int, min_expectancy: float = 0.001) -> str:
        """
        Get trading signal based on regime analysis.
        
        Args:
            regime: Current regime
            min_expectancy: Minimum expectancy threshold
            
        Returns:
            Trading signal: 'trade', 'reduce', or 'halt'
        """
        if not self.regime_gater.should_trade_in_regime(regime, min_expectancy):
            return 'halt'
        
        expectancy = self.regime_gater.get_regime_expectancy(regime)
        
        if expectancy > min_expectancy * 2:
            return 'trade'
        elif expectancy > min_expectancy:
            return 'reduce'
        else:
            return 'halt'
