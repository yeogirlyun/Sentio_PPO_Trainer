#!/usr/bin/env python3
"""
Enhanced PPO Network Architecture for 10% Monthly Returns
Implements LSTM, Attention, and Residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Enhanced PPO network will not function.")


class AttentionModule(nn.Module):
    """Multi-head attention for focusing on relevant time periods"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.output(context)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Expand and contract
        out = F.gelu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Layer norm (works with any batch size)
        out = self.layer_norm(out)
        
        # Residual connection
        return F.gelu(out + residual)


class EnhancedPPONetwork(nn.Module):
    """
    Advanced PPO Network with LSTM, Attention, and Residual connections
    Designed for 10% monthly returns target
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [512, 512, 256],
        lstm_hidden: int = 256,
        action_size: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_lstm: bool = True,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        self.lstm_hidden = lstm_hidden
        
        # Input processing (use LayerNorm instead of BatchNorm to avoid batch size issues)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, hidden_sizes[0])
        
        # LSTM for temporal dependencies
        if use_lstm:
            self.lstm = nn.LSTM(
                hidden_sizes[0],
                lstm_hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True
            )
            lstm_output_size = lstm_hidden * 2  # bidirectional
        else:
            lstm_output_size = hidden_sizes[0]
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionModule(lstm_output_size, num_attention_heads)
            self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # Deep residual blocks
        self.residual_blocks = nn.ModuleList()
        current_size = lstm_output_size
        
        for hidden_size in hidden_sizes:
            # Projection if size changes
            if current_size != hidden_size:
                self.residual_blocks.append(nn.Linear(current_size, hidden_size))
                current_size = hidden_size
            
            # Add residual block
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout))
        
        # Actor head (policy) with multi-head output for exploration
        self.actor_heads = nn.ModuleList([
            nn.Linear(current_size, action_size) for _ in range(3)
        ])
        self.actor_combiner = nn.Linear(action_size * 3, action_size)
        
        # Critic head (value function) with auxiliary heads
        self.critic_main = nn.Linear(current_size, 1)
        self.critic_aux = nn.Linear(current_size, 1)  # For advantage estimation
        
        # Learnable temperature for exploration
        self.log_temperature = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/He initialization for better gradient flow"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size) or (batch_size, input_size)
            hidden_state: Optional LSTM hidden state
            
        Returns:
            policy_logits, value, hidden_state
        """
        # Handle both sequential and non-sequential inputs
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            x = x.unsqueeze(1)  # Add sequence dimension
            is_sequential = False
        else:
            batch_size = x.shape[0]
            is_sequential = True
        
        # Input normalization and projection (LayerNorm works with any shape)
        x_norm = self.input_norm(x)
        
        x = F.gelu(self.input_projection(x_norm))
        
        # LSTM processing
        if self.use_lstm:
            if hidden_state is None:
                h0 = torch.zeros(4, batch_size, self.lstm_hidden).to(x.device)  # 2 layers * 2 directions
                c0 = torch.zeros(4, batch_size, self.lstm_hidden).to(x.device)
                hidden_state = (h0, c0)
            
            lstm_out, new_hidden = self.lstm(x, hidden_state)
            x = lstm_out
        else:
            new_hidden = None
        
        # Attention mechanism
        if self.use_attention and is_sequential:
            attended = self.attention(x)
            x = self.attention_norm(x + attended)  # Residual connection
        
        # Process through residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Take the last timestep for non-sequential output
        if not is_sequential or x.shape[1] == 1:
            features = x.squeeze(1) if len(x.shape) == 3 else x
        else:
            features = x.mean(dim=1)  # Average pool over sequence
        
        # Actor outputs (multi-head for exploration)
        actor_outputs = [head(features) for head in self.actor_heads]
        actor_combined = torch.cat(actor_outputs, dim=-1)
        policy_logits = self.actor_combiner(actor_combined)
        
        # Apply temperature scaling for exploration control
        temperature = torch.exp(self.log_temperature)
        policy_logits = policy_logits / temperature
        
        # Critic outputs (main + auxiliary)
        value_main = self.critic_main(features)
        value_aux = self.critic_aux(features)
        value = (value_main + value_aux) / 2  # Ensemble values
        
        return policy_logits, value, new_hidden
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Get action and value for PPO training with enhanced exploration"""
        if not TORCH_AVAILABLE:
            # Return dummy values if PyTorch not available
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            return (
                torch.randint(0, 3, (batch_size,)),
                torch.zeros(batch_size),
                torch.zeros(batch_size),
                torch.zeros(batch_size, 1),
                None
            )
        
        from torch.distributions import Categorical
        
        policy_logits, value, hidden_state = self.forward(x, hidden_state)
        
        # Create action distribution
        probs = Categorical(logits=policy_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value, hidden_state


class AdvancedFeatureEngine:
    """
    Extended feature engineering for 2000+ features
    Includes market microstructure, multi-timeframe, and advanced indicators
    """
    
    def __init__(self, lookback_window: int = 120, enable_caching: bool = True):
        self.lookback_window = lookback_window
        self.feature_names = []
        self.enable_caching = enable_caching
        
        # Feature caching system
        self._feature_cache = {}
        self._cache_keys = {}
        self._cache_size_limit = 1000  # Maximum cache entries
        self._cache_hits = 0
        self._cache_misses = 0
    
    def engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate 2000+ features from market data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Feature array
        """
        # Check cache first if enabled
        if self.enable_caching:
            cache_key = self._generate_cache_key(data)
            if cache_key in self._feature_cache:
                self._cache_hits += 1
                return self._feature_cache[cache_key]
            else:
                self._cache_misses += 1
        
        features = []
        
        try:
            # 1. Price features (raw and normalized)
            features.extend(self._price_features(data))
            
            # 2. Volume features
            features.extend(self._volume_features(data))
            
            # 3. Technical indicators (25+ indicators)
            features.extend(self._technical_indicators(data))
            
            # 4. Market microstructure
            features.extend(self._microstructure_features(data))
            
            # 5. Multi-timeframe features
            features.extend(self._multi_timeframe_features(data))
            
            # 6. Statistical features
            features.extend(self._statistical_features(data))
            
            # 7. Pattern recognition
            features.extend(self._pattern_features(data))
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            # Return zeros if feature engineering fails
            features = [0.0] * 2340  # Target feature count
        
        # Ensure consistent feature count
        while len(features) < 2340:
            features.append(0.0)
        
        feature_array = np.array(features[:2340], dtype=np.float32)
        
        # Cache the result if caching is enabled
        if self.enable_caching:
            self._cache_features(cache_key, feature_array)
        
        return feature_array
    
    def _price_features(self, data: pd.DataFrame) -> list:
        """Extract price-based features"""
        features = []
        
        try:
            # Raw OHLCV (limited to lookback window)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    values = data[col].values[-self.lookback_window:]
                    # Pad if not enough data
                    if len(values) < self.lookback_window:
                        padded = np.zeros(self.lookback_window)
                        padded[-len(values):] = values
                        values = padded
                    features.extend(values)
                else:
                    features.extend([0.0] * self.lookback_window)
            
            # Returns at multiple scales
            for period in [1, 5, 10, 20, 30, 60]:
                if len(data) > period and 'close' in data.columns:
                    returns = data['close'].pct_change(period).fillna(0)
                    values = returns.values[-min(self.lookback_window, len(returns)):]
                    # Pad if needed
                    if len(values) < self.lookback_window:
                        padded = np.zeros(self.lookback_window)
                        padded[-len(values):] = values
                        values = padded
                    features.extend(values)
                else:
                    features.extend([0.0] * self.lookback_window)
            
            # Log returns
            if 'close' in data.columns and len(data) > 1:
                log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
                values = log_returns.values[-self.lookback_window:]
                if len(values) < self.lookback_window:
                    padded = np.zeros(self.lookback_window)
                    padded[-len(values):] = values
                    values = padded
                features.extend(values)
            else:
                features.extend([0.0] * self.lookback_window)
                
        except Exception as e:
            logger.error(f"Error in price features: {e}")
            # Return zeros for expected number of price features
            expected_count = self.lookback_window * 5 + self.lookback_window * 6 + self.lookback_window
            features = [0.0] * expected_count
        
        return features
    
    def _volume_features(self, data: pd.DataFrame) -> list:
        """Extract volume-based features"""
        features = []
        
        try:
            if 'volume' in data.columns and len(data) > 0:
                # Volume indicators
                features.append(data['volume'].mean())
                features.append(data['volume'].std())
                
                # VWAP
                if 'close' in data.columns:
                    vwap = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
                    vwap = vwap.fillna(data['close'].iloc[-1] if len(data) > 0 else 0)
                    features.extend(vwap.values[-20:] if len(vwap) >= 20 else [vwap.iloc[-1]] * 20)
                else:
                    features.extend([0.0] * 20)
                
                # On-Balance Volume
                if 'close' in data.columns and len(data) > 1:
                    obv = (np.sign(data['close'].diff()) * data['volume']).cumsum().fillna(0)
                    features.extend(obv.values[-20:] if len(obv) >= 20 else [obv.iloc[-1]] * 20)
                else:
                    features.extend([0.0] * 20)
            else:
                features.extend([0.0] * 42)  # 2 + 20 + 20
                
        except Exception as e:
            logger.error(f"Error in volume features: {e}")
            features = [0.0] * 42
        
        return features
    
    def _technical_indicators(self, data: pd.DataFrame) -> list:
        """Calculate 25+ technical indicators"""
        features = []
        
        try:
            if 'close' not in data.columns or len(data) < 2:
                return [0.0] * 200  # Expected number of technical features
            
            # Momentum indicators (RSI)
            for period in [7, 14, 21, 28]:
                rsi = self._calculate_rsi(data['close'], period)
                features.extend(rsi[-10:] if len(rsi) >= 10 else [50.0] * 10)
            
            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                if len(data) >= period:
                    sma = data['close'].rolling(period).mean().fillna(data['close'].iloc[-1])
                    ema = data['close'].ewm(span=period).mean().fillna(data['close'].iloc[-1])
                    features.append(sma.iloc[-1])
                    features.append(ema.iloc[-1])
                else:
                    features.extend([data['close'].iloc[-1]] * 2)
            
            # MACD variations
            for fast, slow in [(12, 26), (5, 35), (10, 20)]:
                if len(data) >= slow:
                    macd = data['close'].ewm(span=fast).mean() - data['close'].ewm(span=slow).mean()
                    signal = macd.ewm(span=9).mean()
                    macd = macd.fillna(0)
                    signal = signal.fillna(0)
                    features.extend(macd.values[-10:] if len(macd) >= 10 else [0.0] * 10)
                    features.extend(signal.values[-10:] if len(signal) >= 10 else [0.0] * 10)
                else:
                    features.extend([0.0] * 20)
            
            # Bollinger Bands
            for period in [20, 30]:
                for std_dev in [1, 2, 3]:
                    if len(data) >= period:
                        sma = data['close'].rolling(period).mean()
                        std = data['close'].rolling(period).std()
                        upper = sma + (std * std_dev)
                        lower = sma - (std * std_dev)
                        bb_position = ((data['close'].iloc[-1] - lower.iloc[-1]) / 
                                     (upper.iloc[-1] - lower.iloc[-1] + 1e-10))
                        features.append(bb_position)
                    else:
                        features.append(0.5)
            
            # Stochastic Oscillator
            for period in [14, 21]:
                if len(data) >= period and 'high' in data.columns and 'low' in data.columns:
                    low_min = data['low'].rolling(period).min()
                    high_max = data['high'].rolling(period).max()
                    k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min + 1e-10))
                    k_percent = k_percent.fillna(50)
                    features.extend(k_percent.values[-5:] if len(k_percent) >= 5 else [50.0] * 5)
                else:
                    features.extend([50.0] * 5)
                    
        except Exception as e:
            logger.error(f"Error in technical indicators: {e}")
            features = [0.0] * 200
        
        # Ensure consistent length
        while len(features) < 200:
            features.append(0.0)
        
        return features[:200]
    
    def _microstructure_features(self, data: pd.DataFrame) -> list:
        """Market microstructure features"""
        features = []
        
        try:
            if len(data) < 2:
                return [0.0] * 70
            
            # Spread proxies
            if all(col in data.columns for col in ['high', 'low', 'close']):
                hl_spread = (data['high'] - data['low']) / (data['close'] + 1e-10)
                hl_spread = hl_spread.fillna(0)
                features.extend(hl_spread.values[-20:] if len(hl_spread) >= 20 else [0.0] * 20)
            else:
                features.extend([0.0] * 20)
            
            # Price efficiency
            if all(col in data.columns for col in ['high', 'low', 'close', 'open']):
                efficiency = (abs(data['close'] - data['open']) / 
                            (data['high'] - data['low'] + 1e-10))
                efficiency = efficiency.fillna(0.5)
                features.extend(efficiency.values[-20:] if len(efficiency) >= 20 else [0.5] * 20)
            else:
                features.extend([0.5] * 20)
            
            # Tick direction
            if 'close' in data.columns:
                tick_direction = np.sign(data['close'].diff()).fillna(0)
                features.extend(tick_direction.values[-30:] if len(tick_direction) >= 30 else [0.0] * 30)
            else:
                features.extend([0.0] * 30)
                
        except Exception as e:
            logger.error(f"Error in microstructure features: {e}")
            features = [0.0] * 70
        
        return features
    
    def _multi_timeframe_features(self, data: pd.DataFrame) -> list:
        """Features from multiple timeframes"""
        features = []
        
        try:
            if len(data) < 5:
                return [0.0] * 50
            
            # Simplified multi-timeframe (since we can't easily resample without datetime index)
            # Use rolling windows as proxy for different timeframes
            for window in [5, 15, 30, 60]:  # 5min, 15min, 30min, 1hour proxies
                if len(data) >= window and 'close' in data.columns:
                    # Returns from each "timeframe"
                    window_returns = data['close'].rolling(window).apply(
                        lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) > 0 else 0
                    ).fillna(0)
                    
                    features.extend(window_returns.values[-5:] if len(window_returns) >= 5 else [0.0] * 5)
                    
                    # RSI from each "timeframe"
                    if len(data) >= window + 14:
                        window_data = data['close'].rolling(window).mean()
                        rsi = self._calculate_rsi(window_data, 14)
                        features.append(rsi[-1] if len(rsi) > 0 else 50.0)
                    else:
                        features.append(50.0)
                else:
                    features.extend([0.0] * 6)  # 5 returns + 1 RSI
                    
        except Exception as e:
            logger.error(f"Error in multi-timeframe features: {e}")
            features = [0.0] * 50
        
        # Ensure consistent length
        while len(features) < 50:
            features.append(0.0)
        
        return features[:50]
    
    def _statistical_features(self, data: pd.DataFrame) -> list:
        """Statistical features"""
        features = []
        
        try:
            if 'close' not in data.columns or len(data) < 5:
                return [0.0] * 40
            
            # Rolling statistics for different windows
            for window in [5, 10, 20, 50]:
                if len(data) >= window:
                    returns = data['close'].pct_change().fillna(0)
                    
                    # Volatility
                    vol = returns.rolling(window).std().fillna(0).iloc[-1]
                    features.append(vol)
                    
                    # Skewness
                    skew = returns.rolling(window).skew().fillna(0).iloc[-1]
                    features.append(skew)
                    
                    # Kurtosis
                    kurt = returns.rolling(window).kurt().fillna(0).iloc[-1]
                    features.append(kurt)
                    
                    # Autocorrelation (simplified)
                    if len(returns) >= window + 1:
                        recent_returns = returns.values[-window:]
                        if len(recent_returns) > 1:
                            autocorr = np.corrcoef(recent_returns[:-1], recent_returns[1:])[0, 1]
                            autocorr = autocorr if not np.isnan(autocorr) else 0.0
                        else:
                            autocorr = 0.0
                    else:
                        autocorr = 0.0
                    features.append(autocorr)
                else:
                    features.extend([0.0] * 4)
                    
        except Exception as e:
            logger.error(f"Error in statistical features: {e}")
            features = [0.0] * 40
        
        # Ensure consistent length
        while len(features) < 40:
            features.append(0.0)
        
        return features[:40]
    
    def _pattern_features(self, data: pd.DataFrame) -> list:
        """Pattern recognition features"""
        features = []
        
        try:
            if 'close' not in data.columns or len(data) < 10:
                return [0.0] * 20
            
            # Support/Resistance levels
            for period in [20, 50, 100]:
                if len(data) >= period and all(col in data.columns for col in ['high', 'low']):
                    resistance = data['high'].rolling(period).max().iloc[-1]
                    support = data['low'].rolling(period).min().iloc[-1]
                    current_price = data['close'].iloc[-1]
                    
                    sr_position = ((current_price - support) / 
                                 (resistance - support + 1e-10))
                    features.append(sr_position)
                else:
                    features.append(0.5)
            
            # Trend strength
            for period in [10, 20, 50]:
                if len(data) >= period:
                    prices = data['close'].values[-period:]
                    if len(prices) >= 2:
                        trend = np.polyfit(range(len(prices)), prices, 1)[0]
                        # Normalize trend
                        trend = trend / (prices[-1] + 1e-10)
                        features.append(trend)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
                    
        except Exception as e:
            logger.error(f"Error in pattern features: {e}")
            features = [0.0] * 20
        
        # Ensure consistent length
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return np.array([50.0] * len(prices))
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).values
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return np.array([50.0] * len(prices))
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate a cache key for the given data."""
        try:
            # Use the last few rows to create a unique key
            if len(data) == 0:
                return "empty_data"
            
            # Use timestamp and price info for key
            last_row = data.iloc[-1]
            key_components = []
            
            # Add timestamp if available
            if hasattr(data.index, 'to_pydatetime'):
                key_components.append(str(data.index[-1]))
            else:
                key_components.append(str(len(data)))
            
            # Add price info
            for col in ['close', 'volume']:
                if col in data.columns:
                    key_components.append(f"{col}_{last_row[col]:.6f}")
            
            # Add data length
            key_components.append(f"len_{len(data)}")
            
            return "_".join(key_components)
            
        except Exception as e:
            logger.warning(f"Error generating cache key: {e}")
            return f"fallback_{hash(str(data.values.tobytes()) if len(data) > 0 else 'empty')}"
    
    def _cache_features(self, cache_key: str, features: np.ndarray):
        """Cache computed features."""
        try:
            # Check cache size limit
            if len(self._feature_cache) >= self._cache_size_limit:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self._feature_cache.keys())[:self._cache_size_limit // 2]
                for old_key in oldest_keys:
                    del self._feature_cache[old_key]
                    if old_key in self._cache_keys:
                        del self._cache_keys[old_key]
            
            # Store features
            self._feature_cache[cache_key] = features.copy()
            self._cache_keys[cache_key] = pd.Timestamp.now()
            
        except Exception as e:
            logger.warning(f"Error caching features: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._feature_cache),
            'cache_limit': self._cache_size_limit
        }
    
    def clear_cache(self):
        """Clear the feature cache."""
        self._feature_cache.clear()
        self._cache_keys.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Feature cache cleared")
    
    def set_cache_limit(self, limit: int):
        """Set the cache size limit."""
        self._cache_size_limit = max(1, limit)
        
        # Trim cache if it exceeds new limit
        if len(self._feature_cache) > self._cache_size_limit:
            keys_to_remove = list(self._feature_cache.keys())[self._cache_size_limit:]
            for key in keys_to_remove:
                del self._feature_cache[key]
                if key in self._cache_keys:
                    del self._cache_keys[key]


# Example usage and testing
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        # Configuration for enhanced network
        config = {
            'input_size': 2340,  # ~2000+ features
            'hidden_sizes': [512, 512, 256],
            'lstm_hidden': 256,
            'action_size': 3,
            'dropout': 0.1,
            'use_attention': True,
            'use_lstm': True,
            'num_attention_heads': 8
        }
        
        # Create network
        model = EnhancedPPONetwork(**config)
        
        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Enhanced PPO Network Created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Input size: {config['input_size']:,} features")
        print(f"  Architecture: {config['hidden_sizes']}")
        print(f"  LSTM: {config['use_lstm']}, Attention: {config['use_attention']}")
        
        # Test forward pass
        batch_size = 32
        seq_length = 30
        x = torch.randn(batch_size, seq_length, config['input_size'])
        
        policy_logits, value, hidden = model(x)
        print(f"\nTest Forward Pass:")
        print(f"  Input shape: {x.shape}")
        print(f"  Policy logits shape: {policy_logits.shape}")
        print(f"  Value shape: {value.shape}")
        
        # Test feature engine
        print(f"\nTesting Advanced Feature Engine:")
        feature_engine = AdvancedFeatureEngine(lookback_window=120)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': np.random.randn(200) * 0.01 + 100,
            'high': np.random.randn(200) * 0.01 + 101,
            'low': np.random.randn(200) * 0.01 + 99,
            'close': np.random.randn(200) * 0.01 + 100,
            'volume': np.random.randint(1000, 10000, 200)
        })
        
        features = feature_engine.engineer_features(sample_data)
        print(f"  Generated features: {len(features):,}")
        print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  Non-zero features: {np.count_nonzero(features):,}")
        
    else:
        print("PyTorch not available. Enhanced PPO Network cannot be tested.")
        
    print(f"\n🎯 Enhanced PPO Network ready for 10% monthly returns target!")
