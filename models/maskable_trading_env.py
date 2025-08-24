#!/usr/bin/env python3
"""
Maskable Trading Environment for PPO
Enhanced trading environment with action masking capabilities for better constraint handling.

Features:
- Dynamic action masking based on position, cash, and risk limits
- Built-in risk management through mask enforcement
- Compatible with sb3-contrib MaskablePPO
- Comprehensive observation space with market indicators
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MaskableTradingEnv(gym.Env):
    """
    Enhanced trading environment with action masking for Maskable PPO.
    
    Action Space: [HOLD, BUY, SELL]
    - HOLD (0): Do nothing
    - BUY (1): Buy shares (long position)
    - SELL (2): Sell shares (short position or close long)
    
    Observation Space: Market data + portfolio state + technical indicators
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 100000,
                 max_position: int = 1000,
                 transaction_cost: float = 0.001,
                 max_drawdown_limit: float = 0.20,
                 lookback_window: int = 50,
                 risk_free_rate: float = 0.02):
        
        super().__init__()
        
        # Environment parameters
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.max_drawdown_limit = max_drawdown_limit
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        
        # Action space: [HOLD, BUY, SELL]
        self.action_space = spaces.Discrete(3)
        
        # Calculate observation space size
        market_features = 5  # OHLCV
        technical_indicators = 10  # RSI, MACD, BB, etc.
        portfolio_state = 6  # cash, position, value, drawdown, etc.
        lookback_features = 3  # price, volume, returns over lookback window
        
        obs_size = market_features + technical_indicators + portfolio_state + (lookback_features * lookback_window)
        
        # Store for validation
        self._expected_obs_size = obs_size
        self._obs_breakdown = {
            'market_features': market_features,
            'technical_indicators': technical_indicators,
            'portfolio_state': portfolio_state,
            'lookback_features': lookback_features * lookback_window
        }
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Portfolio state
        self.cash = self.initial_capital
        self.position = 0  # Number of shares held
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        
        # Trading state
        self.current_step = self.lookback_window
        self.done = False
        self.total_trades = 0
        self.winning_trades = 0
        
        # Performance tracking
        self.portfolio_history = [self.initial_capital]
        self.action_history = []
        self.trade_history = []
        
        # Calculate initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step"""
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Get action mask to check if action is valid
        action_mask = self.get_action_mask()
        
        # Initialize reward
        reward = 0.0
        info = {}
        
        # Check if action is masked (invalid)
        if not action_mask[action]:
            # Small penalty for invalid action (0.01% not 1%)
            reward = -0.0001  # Much smaller penalty
            info['invalid_action'] = True
            logger.warning(f"Invalid action {action} attempted at step {self.current_step}")
        else:
            # Execute valid action
            reward, trade_info = self._execute_action(action, current_price)
            info.update(trade_info)
        
        # Update portfolio value
        self.portfolio_value = self.cash + (self.position * current_price)
        self.portfolio_history.append(self.portfolio_value)
        
        # Update max portfolio value for drawdown calculation
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        # Check termination conditions
        self.done = self._check_termination()
        
        # Move to next step
        self.current_step += 1
        
        # Check if we've reached end of data
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Get next observation
        observation = self._get_observation()
        
        # Add performance metrics to info
        info.update(self._get_performance_metrics())
        
        return observation, reward, self.done, False, info
    
    def get_action_mask(self) -> np.ndarray:
        """
        Return boolean mask for valid actions based on current state.
        
        Returns:
            np.ndarray: Boolean mask [can_hold, can_buy, can_sell]
        """
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Initialize mask - all actions valid by default
        mask = np.array([True, True, True], dtype=bool)
        
        # HOLD is always valid (action 0)
        # mask[0] = True  # Already set
        
        # BUY constraints (action 1)
        buy_cost = current_price * (1 + self.transaction_cost)
        
        # Can't buy if:
        # 1. Insufficient cash
        # 2. Already at maximum long position
        # 3. Would exceed risk limits
        if (self.cash < buy_cost or 
            self.position >= self.max_position or
            self._would_exceed_risk_limits('BUY', current_price)):
            mask[1] = False
        
        # SELL constraints (action 2)
        # Can't sell if:
        # 1. No position to sell (for long-only strategy)
        # 2. Already at maximum short position
        # 3. Would exceed risk limits
        if (self.position <= 0 or
            self.position <= -self.max_position or
            self._would_exceed_risk_limits('SELL', current_price)):
            mask[2] = False
        
        return mask
    
    def _execute_action(self, action: int, current_price: float) -> Tuple[float, Dict]:
        """Execute the given action and return reward and trade info"""
        
        reward = 0.0
        trade_info = {'trade_executed': False, 'trade_type': None, 'trade_amount': 0}
        
        if action == 0:  # HOLD
            # Calculate holding reward based on portfolio value change
            if self.current_step > 0 and len(self.portfolio_history) > 0:
                prev_portfolio = self.portfolio_history[-1]
                current_portfolio = self.cash + (self.position * current_price)
                reward = (current_portfolio - prev_portfolio) / prev_portfolio if prev_portfolio > 0 else 0.0
            else:
                reward = 0.0
            
        elif action == 1:  # BUY
            # Calculate how many shares we can buy (all-or-nothing for QQQ)
            available_cash = self.cash
            cost_per_share = current_price * (1 + self.transaction_cost)
            shares_to_buy = min(
                int(available_cash / cost_per_share),
                self.max_position - self.position
            )
            
            if shares_to_buy > 0:
                # Store portfolio value before trade
                portfolio_before = self.cash + (self.position * current_price)
                
                # Execute buy order
                total_cost = shares_to_buy * cost_per_share
                self.cash -= total_cost
                self.position += shares_to_buy
                self.total_trades += 1
                
                # Calculate portfolio value after trade (using current price)
                portfolio_after = self.cash + (self.position * current_price)
                
                # Reward based on actual portfolio change
                reward = (portfolio_after - portfolio_before) / portfolio_before
                
                trade_info.update({
                    'trade_executed': True,
                    'trade_type': 'BUY',
                    'trade_amount': shares_to_buy,
                    'trade_price': current_price,
                    'trade_cost': total_cost
                })
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': total_cost
                })
        
        elif action == 2:  # SELL
            # Sell all position (all-or-nothing for QQQ)
            shares_to_sell = min(self.position, abs(self.max_position))
            
            if shares_to_sell > 0:
                # Store portfolio value before trade
                portfolio_before = self.cash + (self.position * current_price)
                
                # Execute sell order
                revenue_per_share = current_price * (1 - self.transaction_cost)
                total_revenue = shares_to_sell * revenue_per_share
                
                self.cash += total_revenue
                self.position -= shares_to_sell
                self.total_trades += 1
                
                # Calculate portfolio value after trade (using current price)
                portfolio_after = self.cash + (self.position * current_price)
                
                # Reward based on actual portfolio change
                reward = (portfolio_after - portfolio_before) / portfolio_before
                
                # Track winning trades
                if reward > 0:
                    self.winning_trades += 1
                
                trade_info.update({
                    'trade_executed': True,
                    'trade_type': 'SELL',
                    'trade_amount': shares_to_sell,
                    'trade_price': current_price,
                    'trade_revenue': total_revenue,
                    'pnl': pnl if 'pnl' in locals() else 0
                })
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'revenue': total_revenue
                })
        
        # Add portfolio change reward
        if len(self.portfolio_history) > 1:
            portfolio_change = (self.portfolio_value - self.portfolio_history[-1]) / self.portfolio_history[-1]
            reward += portfolio_change * 10  # Scale portfolio change reward
        
        # Clip reward to reasonable range
        reward = np.clip(reward, -0.1, 0.1)
        
        return reward, trade_info
    
    def _would_exceed_risk_limits(self, action: str, current_price: float) -> bool:
        """Check if action would exceed risk management limits"""
        
        # Calculate potential portfolio value after action
        if action == 'BUY':
            potential_position = self.position + 1
        elif action == 'SELL':
            potential_position = self.position - 1
        else:
            return False
        
        potential_portfolio_value = self.cash + (potential_position * current_price)
        
        # Check drawdown limit
        if self.max_portfolio_value > 0:
            potential_drawdown = (self.max_portfolio_value - potential_portfolio_value) / self.max_portfolio_value
            if potential_drawdown > self.max_drawdown_limit:
                return True
        
        # Check position concentration (don't put more than 95% in single position)
        position_value = abs(potential_position * current_price)
        if position_value > 0.95 * potential_portfolio_value:
            return True
        
        return False
    
    def _calculate_average_buy_price(self) -> float:
        """Calculate average buy price from trade history"""
        buy_trades = [trade for trade in self.trade_history if trade['action'] == 'BUY']
        if not buy_trades:
            return 0.0
        
        total_cost = sum(trade['cost'] for trade in buy_trades)
        total_shares = sum(trade['shares'] for trade in buy_trades)
        
        return total_cost / total_shares if total_shares > 0 else 0.0
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        
        # Terminate if portfolio value drops too much
        if self.portfolio_value < 0.1 * self.initial_capital:
            return True
        
        # Terminate if maximum drawdown exceeded
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if drawdown > self.max_drawdown_limit:
                return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Generate current observation vector"""
        
        current_data = self.data.iloc[self.current_step]
        
        # 1. Current market data (5 features)
        market_features = [
            current_data['open'],
            current_data['high'], 
            current_data['low'],
            current_data['close'],
            current_data['volume']
        ]
        
        # 2. Technical indicators (10 features)
        tech_indicators = self._calculate_technical_indicators()
        
        # 3. Portfolio state (6 features)
        portfolio_features = [
            self.cash / self.initial_capital,  # Normalized cash
            self.position / self.max_position,  # Normalized position
            self.portfolio_value / self.initial_capital,  # Normalized portfolio value
            (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0,  # Drawdown
            self.total_trades / 100,  # Normalized trade count
            self.winning_trades / max(self.total_trades, 1)  # Win rate
        ]
        
        # 4. Lookback window features (3 * lookback_window features)
        lookback_features = self._get_lookback_features()
        
        # Combine all features
        observation = np.array(
            market_features + tech_indicators + portfolio_features + lookback_features,
            dtype=np.float32
        )
        
        # Handle any NaN or infinite values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Validate observation size
        self._validate_observation_size(observation)
        
        return observation
    
    def _calculate_technical_indicators(self) -> List[float]:
        """Calculate technical indicators for current observation"""
        
        # Get recent data for indicator calculation
        end_idx = self.current_step + 1
        start_idx = max(0, end_idx - 50)  # Use up to 50 periods for indicators
        recent_data = self.data.iloc[start_idx:end_idx]
        
        indicators = []
        
        try:
            # RSI (14-period)
            rsi = self._calculate_rsi(recent_data['close'], 14)
            indicators.append(rsi / 100.0)  # Normalize to 0-1
            
            # MACD
            macd, signal = self._calculate_macd(recent_data['close'])
            indicators.extend([macd / 100.0, signal / 100.0])  # Normalize
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(recent_data['close'], 20, 2)
            current_price = recent_data['close'].iloc[-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            indicators.append(bb_position)
            
            # Moving averages
            ma_5 = recent_data['close'].rolling(5).mean().iloc[-1]
            ma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
            ma_ratio = ma_5 / ma_20 if ma_20 > 0 else 1.0
            indicators.append(ma_ratio - 1.0)  # Center around 0
            
            # Volume indicators
            volume_ma = recent_data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = recent_data['volume'].iloc[-1] / volume_ma if volume_ma > 0 else 1.0
            indicators.append(np.log(volume_ratio))  # Log transform
            
            # Price momentum
            price_change_1d = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-2] - 1) if len(recent_data) > 1 else 0
            price_change_5d = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-6] - 1) if len(recent_data) > 5 else 0
            indicators.extend([price_change_1d * 100, price_change_5d * 100])  # Convert to percentage
            
            # Volatility (20-period)
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0
            indicators.append(volatility * 100)  # Convert to percentage
            
            # ATR (Average True Range)
            atr = self._calculate_atr(recent_data, 14)
            indicators.append(atr / recent_data['close'].iloc[-1])  # Normalize by price
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            # Fill with zeros if calculation fails
            indicators = [0.0] * 10
        
        # Ensure we have exactly 10 indicators
        while len(indicators) < 10:
            indicators.append(0.0)
        
        return indicators[:10]
    
    def _get_lookback_features(self) -> List[float]:
        """Get lookback window features (price, volume, returns)"""
        
        features = []
        
        end_idx = self.current_step + 1
        start_idx = max(0, end_idx - self.lookback_window)
        lookback_data = self.data.iloc[start_idx:end_idx]
        
        # Normalize prices by current price
        current_price = self.data.iloc[self.current_step]['close']
        
        for i in range(self.lookback_window):
            if i < len(lookback_data):
                row = lookback_data.iloc[-(i+1)]  # Reverse order (most recent first)
                
                # Normalized price
                price_norm = row['close'] / current_price - 1.0
                features.append(price_norm)
                
                # Normalized volume
                volume_norm = np.log(row['volume'] / lookback_data['volume'].mean()) if lookback_data['volume'].mean() > 0 else 0
                features.append(volume_norm)
                
                # Returns
                if i < len(lookback_data) - 1:
                    prev_price = lookback_data.iloc[-(i+2)]['close']
                    returns = (row['close'] / prev_price - 1.0) * 100  # Percentage returns
                else:
                    returns = 0.0
                features.append(returns)
            else:
                # Pad with zeros if not enough data
                features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _validate_observation_size(self, observation: np.ndarray):
        """Validate that observation size matches expected size."""
        actual_size = len(observation)
        expected_size = self._expected_obs_size
        
        if actual_size != expected_size:
            # Log detailed breakdown for debugging
            logger.error(f"Observation size mismatch!")
            logger.error(f"Expected: {expected_size}, Actual: {actual_size}")
            logger.error(f"Breakdown: {self._obs_breakdown}")
            
            # Try to identify which component is wrong
            current_data = self.data.iloc[self.current_step]
            market_features = [current_data['open'], current_data['high'], current_data['low'], 
                             current_data['close'], current_data['volume']]
            tech_indicators = self._calculate_technical_indicators()
            portfolio_features = [
                self.cash / self.initial_capital,
                self.position / self.max_position,
                self.portfolio_value / self.initial_capital,
                (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0,
                self.total_trades / 100,
                self.winning_trades / max(self.total_trades, 1)
            ]
            lookback_features = self._get_lookback_features()
            
            logger.error(f"Actual sizes - Market: {len(market_features)}, Tech: {len(tech_indicators)}, "
                        f"Portfolio: {len(portfolio_features)}, Lookback: {len(lookback_features)}")
            
            from .exceptions import ObservationSpaceError
            raise ObservationSpaceError(
                expected_shape=(expected_size,),
                actual_shape=(actual_size,)
            )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        return macd.iloc[-1], signal_line.iloc[-1]
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices.iloc[-1]
            return current_price * 1.02, current_price * 0.98, current_price
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band.iloc[-1], lower_band.iloc[-1], sma.iloc[-1]
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(data) < 2:
            return 0.0
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        return atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.0
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics"""
        
        total_return = (self.portfolio_value / self.initial_capital) - 1.0
        
        # Calculate Sharpe ratio
        if len(self.portfolio_history) > 1:
            returns = pd.Series(self.portfolio_history).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                excess_returns = returns.mean() - (self.risk_free_rate / 252)  # Daily risk-free rate
                sharpe_ratio = excess_returns / returns.std() * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(self.portfolio_history)
        drawdown = (peak - self.portfolio_history) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': self.position
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info"""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': self.position,
            'total_trades': self.total_trades
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            current_data = self.data.iloc[self.current_step]
            print(f"Step: {self.current_step}")
            print(f"Price: ${current_data['close']:.2f}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Position: {self.position} shares")
            print(f"Total Return: {((self.portfolio_value / self.initial_capital) - 1) * 100:.2f}%")
            print("-" * 40)

if __name__ == "__main__":
    # Test the environment
    from data.data_manager import load_market_data
    
    print("🧪 Testing Maskable Trading Environment")
    
    # Load test data
    data = load_market_data()
    test_data = data.tail(1000)
    
    # Create environment
    env = MaskableTradingEnv(test_data)
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Environment reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    
    # Test action masking
    mask = env.get_action_mask()
    print(f"✅ Action mask: {mask}")
    
    # Test a few steps
    for i in range(5):
        # Get valid actions
        valid_actions = np.where(mask)[0]
        action = np.random.choice(valid_actions)
        
        obs, reward, done, truncated, info = env.step(action)
        mask = env.get_action_mask()
        
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Done={done}")
        print(f"   Portfolio: ${info.get('portfolio_value', 0):.2f}")
        print(f"   Valid actions: {mask}")
        
        if done:
            break
    
    print("🎉 Environment test completed successfully!")
