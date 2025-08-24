#!/usr/bin/env python3
"""
Dynamic Action Masking for Maskable PPO
Advanced action masking system that adapts to market conditions, risk constraints, and trading rules.

Features:
- Market regime-based action masking
- Volatility-adaptive position constraints
- Risk budget allocation masking
- Liquidity-aware position sizing
- Time-of-day trading restrictions
- Regulatory compliance masking
- Portfolio-level risk management
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_LOW_VOL = 0
    BULL_HIGH_VOL = 1
    BEAR_LOW_VOL = 2
    BEAR_HIGH_VOL = 3
    SIDEWAYS = 4
    CRISIS = 5


class TradingSession(Enum):
    """Trading session classifications."""
    PRE_MARKET = 0
    MARKET_OPEN = 1
    REGULAR_HOURS = 2
    MARKET_CLOSE = 3
    AFTER_HOURS = 4
    CLOSED = 5


@dataclass
class RiskConstraints:
    """Risk constraint configuration."""
    max_position_size: float = 0.1  # Maximum position as fraction of portfolio
    max_daily_loss: float = 0.02    # Maximum daily loss limit
    max_drawdown: float = 0.05      # Maximum drawdown limit
    var_limit: float = 0.03         # Value at Risk limit
    concentration_limit: float = 0.2 # Maximum concentration in single asset
    leverage_limit: float = 2.0     # Maximum leverage allowed


@dataclass
class MarketConditions:
    """Current market condition state."""
    volatility: float
    regime: MarketRegime
    liquidity: float
    spread: float
    volume_ratio: float
    momentum: float
    trend_strength: float
    correlation: float


class DynamicActionMasker:
    """
    Advanced dynamic action masking system for Maskable PPO.
    
    This system provides intelligent action masking based on:
    - Market conditions and regimes
    - Risk management constraints
    - Portfolio state and exposure
    - Regulatory and operational rules
    - Time-based restrictions
    """
    
    def __init__(self,
                 action_space_size: int = 3,
                 risk_constraints: Optional[RiskConstraints] = None,
                 enable_regime_masking: bool = True,
                 enable_risk_masking: bool = True,
                 enable_time_masking: bool = True,
                 enable_liquidity_masking: bool = True):
        """
        Initialize the Dynamic Action Masker.
        
        Args:
            action_space_size: Number of possible actions
            risk_constraints: Risk constraint configuration
            enable_regime_masking: Enable market regime-based masking
            enable_risk_masking: Enable risk-based masking
            enable_time_masking: Enable time-based masking
            enable_liquidity_masking: Enable liquidity-based masking
        """
        self.action_space_size = action_space_size
        self.risk_constraints = risk_constraints or RiskConstraints()
        self.enable_regime_masking = enable_regime_masking
        self.enable_risk_masking = enable_risk_masking
        self.enable_time_masking = enable_time_masking
        self.enable_liquidity_masking = enable_liquidity_masking
        
        # Action mapping (customize based on your action space)
        self.action_mapping = {
            0: "hold",
            1: "buy_small",
            2: "buy_large",
            3: "sell_small",
            4: "sell_large"
        } if action_space_size > 3 else {
            0: "sell",
            1: "hold", 
            2: "buy"
        }
        
        # Historical data for adaptive thresholds
        self.volatility_history = []
        self.regime_history = []
        self.performance_history = []
        
    def extract_market_conditions(self, state: np.ndarray) -> MarketConditions:
        """
        Extract market conditions from the environment state.
        
        Args:
            state: Environment state vector
            
        Returns:
            MarketConditions object with extracted features
        """
        # Assuming state structure: [price_features, technical_indicators, portfolio_state, market_features]
        # Adjust indices based on your actual state representation
        
        try:
            volatility = state[0] if len(state) > 0 else 0.01
            momentum = state[1] if len(state) > 1 else 0.0
            volume_ratio = state[2] if len(state) > 2 else 1.0
            spread = state[3] if len(state) > 3 else 0.001
            trend_strength = state[4] if len(state) > 4 else 0.0
            correlation = state[5] if len(state) > 5 else 0.0
            
            # Determine market regime based on volatility and momentum
            regime = self._classify_market_regime(volatility, momentum, trend_strength)
            
            # Calculate liquidity proxy
            liquidity = min(volume_ratio / max(spread, 1e-6), 100.0)
            
            return MarketConditions(
                volatility=volatility,
                regime=regime,
                liquidity=liquidity,
                spread=spread,
                volume_ratio=volume_ratio,
                momentum=momentum,
                trend_strength=trend_strength,
                correlation=correlation
            )
        except Exception as e:
            logger.warning(f"Error extracting market conditions: {e}")
            return MarketConditions(
                volatility=0.01,
                regime=MarketRegime.SIDEWAYS,
                liquidity=1.0,
                spread=0.001,
                volume_ratio=1.0,
                momentum=0.0,
                trend_strength=0.0,
                correlation=0.0
            )
    
    def _classify_market_regime(self, volatility: float, momentum: float, trend_strength: float) -> MarketRegime:
        """Classify current market regime based on market indicators."""
        vol_threshold_low = 0.01
        vol_threshold_high = 0.03
        momentum_threshold = 0.02
        trend_threshold = 0.5
        
        # Crisis detection (very high volatility)
        if volatility > 0.05:
            return MarketRegime.CRISIS
        
        # Sideways market (low trend strength)
        if trend_strength < trend_threshold:
            return MarketRegime.SIDEWAYS
        
        # Bull/Bear classification based on momentum
        if momentum > momentum_threshold:
            return MarketRegime.BULL_LOW_VOL if volatility < vol_threshold_high else MarketRegime.BULL_HIGH_VOL
        elif momentum < -momentum_threshold:
            return MarketRegime.BEAR_LOW_VOL if volatility < vol_threshold_high else MarketRegime.BEAR_HIGH_VOL
        else:
            return MarketRegime.SIDEWAYS
    
    def _get_trading_session(self, time_of_day: float) -> TradingSession:
        """Determine trading session based on time of day (0-1 normalized)."""
        if time_of_day < 0.1:  # Pre-market
            return TradingSession.PRE_MARKET
        elif time_of_day < 0.15:  # Market open
            return TradingSession.MARKET_OPEN
        elif time_of_day < 0.85:  # Regular hours
            return TradingSession.REGULAR_HOURS
        elif time_of_day < 0.9:  # Market close
            return TradingSession.MARKET_CLOSE
        elif time_of_day < 1.0:  # After hours
            return TradingSession.AFTER_HOURS
        else:  # Market closed
            return TradingSession.CLOSED
    
    def get_action_mask(self,
                       state: np.ndarray,
                       current_position: float,
                       available_capital: float,
                       portfolio_value: float,
                       time_of_day: Optional[float] = None,
                       additional_constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate dynamic action mask based on current conditions.
        
        Args:
            state: Environment state vector
            current_position: Current position size
            available_capital: Available capital for trading
            portfolio_value: Total portfolio value
            time_of_day: Time of day (0-1 normalized)
            additional_constraints: Additional constraint parameters
            
        Returns:
            Boolean mask array (True = action allowed, False = action blocked)
        """
        # Initialize mask (all actions allowed by default)
        mask = np.ones(self.action_space_size, dtype=bool)
        
        # Extract market conditions
        market_conditions = self.extract_market_conditions(state)
        
        # Apply different masking strategies
        if self.enable_regime_masking:
            mask = self._apply_regime_masking(mask, market_conditions, current_position)
        
        if self.enable_risk_masking:
            mask = self._apply_risk_masking(mask, current_position, available_capital, 
                                          portfolio_value, market_conditions)
        
        if self.enable_time_masking and time_of_day is not None:
            mask = self._apply_time_masking(mask, time_of_day)
        
        if self.enable_liquidity_masking:
            mask = self._apply_liquidity_masking(mask, market_conditions)
        
        # Apply additional custom constraints
        if additional_constraints:
            mask = self._apply_additional_constraints(mask, additional_constraints, 
                                                    current_position, market_conditions)
        
        # Ensure at least one action is always available (hold)
        if not mask.any():
            hold_action = 1 if self.action_space_size == 3 else 0
            mask[hold_action] = True
            logger.warning("All actions masked, enabling hold action")
        
        return mask
    
    def _apply_regime_masking(self, mask: np.ndarray, 
                            market_conditions: MarketConditions,
                            current_position: float) -> np.ndarray:
        """Apply market regime-based action masking."""
        regime = market_conditions.regime
        
        if regime == MarketRegime.CRISIS:
            # During crisis, only allow position reduction and holds
            if self.action_space_size == 3:
                if current_position > 0:
                    mask[2] = False  # No more buying
                elif current_position < 0:
                    mask[0] = False  # No more selling
            else:
                mask[1] = False  # No large buys
                mask[4] = False  # No large sells
        
        elif regime == MarketRegime.BEAR_HIGH_VOL:
            # In bear market with high volatility, limit long positions
            if self.action_space_size == 3:
                mask[2] = False  # No buying
            else:
                mask[1] = False  # No small buys
                mask[2] = False  # No large buys
        
        elif regime == MarketRegime.BULL_HIGH_VOL:
            # In bull market with high volatility, limit short positions
            if self.action_space_size == 3:
                if current_position <= 0:
                    mask[0] = False  # No selling if not long
            else:
                mask[3] = False  # No small sells
                mask[4] = False  # No large sells
        
        return mask
    
    def _apply_risk_masking(self, mask: np.ndarray,
                          current_position: float,
                          available_capital: float,
                          portfolio_value: float,
                          market_conditions: MarketConditions) -> np.ndarray:
        """Apply risk-based action masking."""
        position_ratio = abs(current_position) / max(portfolio_value, 1.0)
        
        # Maximum position size constraint
        if position_ratio >= self.risk_constraints.max_position_size:
            if self.action_space_size == 3:
                if current_position > 0:
                    mask[2] = False  # No more buying
                elif current_position < 0:
                    mask[0] = False  # No more selling
            else:
                if current_position > 0:
                    mask[1] = mask[2] = False  # No more buys
                elif current_position < 0:
                    mask[3] = mask[4] = False  # No more sells
        
        # Volatility-based position sizing
        vol_adjustment = min(market_conditions.volatility / 0.02, 2.0)  # Scale by volatility
        adjusted_max_position = self.risk_constraints.max_position_size / vol_adjustment
        
        if position_ratio >= adjusted_max_position:
            # Reduce allowed position increases in high volatility
            if self.action_space_size > 3:
                if current_position > 0:
                    mask[2] = False  # No large buys
                elif current_position < 0:
                    mask[4] = False  # No large sells
        
        # Available capital constraint
        capital_ratio = available_capital / max(portfolio_value, 1.0)
        if capital_ratio < 0.1:  # Less than 10% capital available
            if self.action_space_size == 3:
                mask[2] = False  # No buying
            else:
                mask[1] = mask[2] = False  # No buys
        
        return mask
    
    def _apply_time_masking(self, mask: np.ndarray, time_of_day: float) -> np.ndarray:
        """Apply time-based action masking."""
        session = self._get_trading_session(time_of_day)
        
        if session == TradingSession.CLOSED:
            # Market closed - only allow holds
            mask[:] = False
            hold_action = 1 if self.action_space_size == 3 else 0
            mask[hold_action] = True
        
        elif session in [TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS]:
            # Limited trading in extended hours
            if self.action_space_size > 3:
                mask[2] = mask[4] = False  # No large positions
        
        elif session in [TradingSession.MARKET_OPEN, TradingSession.MARKET_CLOSE]:
            # High volatility periods - be more conservative
            if self.action_space_size > 3:
                mask[2] = mask[4] = False  # No large positions
        
        return mask
    
    def _apply_liquidity_masking(self, mask: np.ndarray, 
                               market_conditions: MarketConditions) -> np.ndarray:
        """Apply liquidity-based action masking."""
        liquidity_threshold = 0.5
        
        if market_conditions.liquidity < liquidity_threshold:
            # Low liquidity - limit large positions
            if self.action_space_size > 3:
                mask[2] = mask[4] = False  # No large trades
        
        # High spread constraint
        if market_conditions.spread > 0.01:  # 1% spread
            if self.action_space_size > 3:
                mask[2] = mask[4] = False  # No large trades in wide spreads
        
        return mask
    
    def _apply_additional_constraints(self, mask: np.ndarray,
                                    constraints: Dict[str, Any],
                                    current_position: float,
                                    market_conditions: MarketConditions) -> np.ndarray:
        """Apply additional custom constraints."""
        
        # Custom volatility threshold
        if 'max_volatility' in constraints:
            if market_conditions.volatility > constraints['max_volatility']:
                # High volatility - only allow position reduction
                if current_position > 0 and self.action_space_size == 3:
                    mask[2] = False
                elif current_position < 0 and self.action_space_size == 3:
                    mask[0] = False
        
        # Custom correlation constraint
        if 'max_correlation' in constraints:
            if abs(market_conditions.correlation) > constraints['max_correlation']:
                # High correlation - reduce position sizing
                if self.action_space_size > 3:
                    mask[2] = mask[4] = False
        
        # Custom momentum constraint
        if 'momentum_threshold' in constraints:
            threshold = constraints['momentum_threshold']
            if abs(market_conditions.momentum) < threshold:
                # Low momentum - avoid large positions
                if self.action_space_size > 3:
                    mask[2] = mask[4] = False
        
        return mask
    
    def update_history(self, market_conditions: MarketConditions, performance: float):
        """Update historical data for adaptive thresholds."""
        self.volatility_history.append(market_conditions.volatility)
        self.regime_history.append(market_conditions.regime)
        self.performance_history.append(performance)
        
        # Keep only recent history
        max_history = 1000
        if len(self.volatility_history) > max_history:
            self.volatility_history = self.volatility_history[-max_history:]
            self.regime_history = self.regime_history[-max_history:]
            self.performance_history = self.performance_history[-max_history:]
    
    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """Calculate adaptive thresholds based on historical performance."""
        if len(self.performance_history) < 10:
            return {}
        
        # Calculate performance by regime
        regime_performance = {}
        for i, regime in enumerate(self.regime_history):
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(self.performance_history[i])
        
        # Calculate adaptive volatility threshold
        vol_performance = list(zip(self.volatility_history, self.performance_history))
        vol_performance.sort(key=lambda x: x[0])  # Sort by volatility
        
        # Find optimal volatility threshold (simplified)
        best_vol_threshold = np.percentile(self.volatility_history, 75)
        
        return {
            'adaptive_vol_threshold': best_vol_threshold,
            'regime_performance': {k: np.mean(v) for k, v in regime_performance.items()}
        }
    
    def get_masking_statistics(self) -> Dict[str, Any]:
        """Get statistics about action masking for analysis."""
        return {
            'total_masks_applied': len(self.volatility_history),
            'regime_distribution': {regime: self.regime_history.count(regime) 
                                  for regime in MarketRegime},
            'avg_volatility': np.mean(self.volatility_history) if self.volatility_history else 0,
            'avg_performance': np.mean(self.performance_history) if self.performance_history else 0
        }


class HierarchicalActionMasker(DynamicActionMasker):
    """
    Hierarchical action masking for multi-level decision making.
    
    Supports hierarchical action spaces:
    - Level 1: Strategy selection (trend following, mean reversion, etc.)
    - Level 2: Position sizing (small, medium, large)
    - Level 3: Timing (immediate, delayed, conditional)
    """
    
    def __init__(self, action_hierarchy: Dict[str, List[str]], **kwargs):
        """
        Initialize hierarchical action masker.
        
        Args:
            action_hierarchy: Dictionary defining action hierarchy
        """
        super().__init__(**kwargs)
        self.action_hierarchy = action_hierarchy
        self.hierarchy_levels = len(action_hierarchy)
    
    def get_hierarchical_mask(self, 
                            state: np.ndarray,
                            current_position: float,
                            **kwargs) -> Dict[str, np.ndarray]:
        """Get masks for each level of the hierarchy."""
        masks = {}
        
        # Get base mask
        base_mask = self.get_action_mask(state, current_position, **kwargs)
        
        # Apply hierarchical constraints
        for level, actions in self.action_hierarchy.items():
            level_mask = np.ones(len(actions), dtype=bool)
            
            # Apply level-specific constraints
            if level == 'strategy':
                level_mask = self._mask_strategy_level(level_mask, state)
            elif level == 'sizing':
                level_mask = self._mask_sizing_level(level_mask, state, current_position)
            elif level == 'timing':
                level_mask = self._mask_timing_level(level_mask, state)
            
            masks[level] = level_mask
        
        return masks
    
    def _mask_strategy_level(self, mask: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Apply strategy-level masking."""
        market_conditions = self.extract_market_conditions(state)
        
        # Example: Disable mean reversion in trending markets
        if market_conditions.trend_strength > 0.7:
            # Assuming index 1 is mean reversion strategy
            if len(mask) > 1:
                mask[1] = False
        
        return mask
    
    def _mask_sizing_level(self, mask: np.ndarray, state: np.ndarray, 
                          current_position: float) -> np.ndarray:
        """Apply position sizing level masking."""
        market_conditions = self.extract_market_conditions(state)
        
        # Disable large positions in high volatility
        if market_conditions.volatility > 0.03 and len(mask) > 2:
            mask[2] = False  # Assuming index 2 is large size
        
        return mask
    
    def _mask_timing_level(self, mask: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Apply timing level masking."""
        market_conditions = self.extract_market_conditions(state)
        
        # Disable delayed execution in high volatility
        if market_conditions.volatility > 0.02 and len(mask) > 1:
            mask[1] = False  # Assuming index 1 is delayed timing
        
        return mask
