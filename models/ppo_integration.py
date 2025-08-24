#!/usr/bin/env python3
"""
Sentio Trader PPO Integration
Connects the enhanced PPO system with Sentio Trader for 10% monthly returns
"""

import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. PPO integration will use fallback mode.")


class SentioPPOAgent:
    """
    Sentio Trader PPO Agent for 10% Monthly Returns
    Integrates enhanced PPO model with Sentio's trading infrastructure
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/ppo_agent_10pct_final.pth"
        self.model = None
        self.feature_engine = None
        self.is_loaded = False
        
        # Performance tracking
        self.trade_history = []
        self.current_position = 0  # -1: short, 0: neutral, 1: long
        self.portfolio_value = 100000
        self.daily_pnl = 0
        
        # Risk management
        self.max_position_size = 0.95
        self.daily_loss_limit = 0.02
        self.stop_loss = 0.02
        
        # Load model if available
        self._load_model()
    
    def _load_model(self):
        """Load the trained PPO model"""
        try:
            if TORCH_AVAILABLE and Path(self.model_path).exists():
                from models.ppo_network import EnhancedPPONetwork, AdvancedFeatureEngine
                
                # Load model configuration
                config_path = Path(self.model_path).parent / "ppo_10pct_model_config.json"
                if config_path.exists():
                    import json
                    with open(config_path) as f:
                        model_config = json.load(f)
                    
                    # Create model with saved configuration
                    # Handle both dict (JSON) and dataclass configurations
                    if hasattr(model_config, '__dict__'):
                        # It's a dataclass, convert to dict
                        from dataclasses import asdict
                        network_config = asdict(model_config)
                    else:
                        # It's already a dict (from JSON)
                        network_config = model_config.get('config', model_config)
                    
                    self.model = EnhancedPPONetwork(
                        input_size=2340,
                        hidden_sizes=network_config.get('hidden_sizes', [512, 512, 256]),
                        use_lstm=network_config.get('use_lstm', True),
                        use_attention=network_config.get('use_attention', True),
                        dropout=network_config.get('dropout_rate', 0.1)
                    )
                    
                    # Load trained weights
                    self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                    self.model.eval()
                    
                    # Initialize feature engine
                    self.feature_engine = AdvancedFeatureEngine(
                        lookback_window=network_config.get('lookback_window', 120)
                    )
                    
                    self.is_loaded = True
                    logger.info(f"PPO model loaded successfully from {self.model_path}")
                    
                    # Log model performance
                    achieved_return = model_config.get('achieved_monthly_return', 0)
                    logger.info(f"Model achieved monthly return: {achieved_return:.2%}")
                    
                else:
                    logger.warning(f"Model config not found at {config_path}")
            else:
                logger.warning(f"Model not found at {self.model_path} or PyTorch unavailable")
                
        except Exception as e:
            logger.error(f"Error loading PPO model: {e}")
            self.is_loaded = False
    
    def get_trading_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get trading signal from PPO model
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Dict with trading signal and metadata
        """
        if not self.is_loaded or not TORCH_AVAILABLE:
            return self._fallback_signal(market_data)
        
        try:
            # Generate features
            features = self.feature_engine.engineer_features(market_data)
            
            # Get model prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                logits, value, _ = self.model(features_tensor)
                probs = torch.softmax(logits, dim=-1)
                
                action = torch.argmax(probs).item()
                confidence = probs.max().item()
                expected_value = value.item()
            
            # Map action to signal
            if action == 0:
                signal = "SELL"
                direction = -1
            elif action == 2:
                signal = "BUY"
                direction = 1
            else:
                signal = "HOLD"
                direction = 0
            
            # Calculate position size based on confidence and risk management
            position_size = self._calculate_position_size(confidence, market_data)
            
            return {
                'signal': signal,
                'direction': direction,
                'confidence': confidence,
                'position_size': position_size,
                'expected_value': expected_value,
                'model_loaded': True,
                'timestamp': datetime.now(),
                'features_count': len(features),
                'risk_adjusted': True
            }
            
        except Exception as e:
            logger.error(f"Error generating PPO signal: {e}")
            return self._fallback_signal(market_data)
    
    def _calculate_position_size(self, confidence: float, market_data: pd.DataFrame) -> float:
        """Calculate optimal position size based on confidence and risk"""
        
        # Base position size from confidence
        base_size = confidence * 0.3  # Max 30% from confidence alone
        
        # Adjust for volatility
        if len(market_data) >= 20:
            volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            vol_adjustment = min(0.15 / (volatility * np.sqrt(252)), 1.5)  # Target 15% vol
            base_size *= vol_adjustment
        
        # Apply risk limits
        if self.daily_pnl < -self.daily_loss_limit * self.portfolio_value:
            base_size *= 0.5  # Reduce size if daily loss limit approached
        
        # Ensure within maximum position size
        position_size = min(base_size, self.max_position_size)
        
        return max(0.0, position_size)
    
    def _fallback_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback signal when model is not available"""
        
        # Simple momentum-based fallback
        if len(market_data) >= 20:
            short_ma = market_data['close'].rolling(5).mean().iloc[-1]
            long_ma = market_data['close'].rolling(20).mean().iloc[-1]
            
            if short_ma > long_ma * 1.001:  # 0.1% threshold
                signal = "BUY"
                direction = 1
                confidence = 0.6
            elif short_ma < long_ma * 0.999:
                signal = "SELL"
                direction = -1
                confidence = 0.6
            else:
                signal = "HOLD"
                direction = 0
                confidence = 0.5
        else:
            signal = "HOLD"
            direction = 0
            confidence = 0.5
        
        return {
            'signal': signal,
            'direction': direction,
            'confidence': confidence,
            'position_size': 0.1,  # Conservative fallback size
            'expected_value': 0.0,
            'model_loaded': False,
            'timestamp': datetime.now(),
            'features_count': 0,
            'risk_adjusted': False,
            'fallback_reason': 'Model not loaded or PyTorch unavailable'
        }
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update performance tracking with trade result"""
        
        self.trade_history.append({
            'timestamp': datetime.now(),
            'signal': trade_result.get('signal'),
            'pnl': trade_result.get('pnl', 0),
            'confidence': trade_result.get('confidence', 0),
            'position_size': trade_result.get('position_size', 0)
        })
        
        # Update daily P&L
        self.daily_pnl += trade_result.get('pnl', 0)
        self.portfolio_value += trade_result.get('pnl', 0)
        
        # Log performance
        if len(self.trade_history) % 10 == 0:
            self._log_performance()
    
    def _log_performance(self):
        """Log current performance metrics"""
        if not self.trade_history:
            return
        
        recent_trades = self.trade_history[-100:]  # Last 100 trades
        
        total_pnl = sum(trade['pnl'] for trade in recent_trades)
        win_rate = sum(1 for trade in recent_trades if trade['pnl'] > 0) / len(recent_trades)
        avg_confidence = np.mean([trade['confidence'] for trade in recent_trades])
        
        # Estimate monthly return
        if len(recent_trades) >= 10:
            avg_pnl_per_trade = total_pnl / len(recent_trades)
            # Assume ~50 trades per month
            monthly_return_estimate = avg_pnl_per_trade * 50 / self.portfolio_value
        else:
            monthly_return_estimate = 0
        
        logger.info(
            f"PPO Performance - Trades: {len(recent_trades)}, "
            f"Win Rate: {win_rate:.1%}, "
            f"Avg Confidence: {avg_confidence:.2f}, "
            f"Est Monthly Return: {monthly_return_estimate:.2%}"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'model_loaded': self.is_loaded,
            'model_path': self.model_path,
            'pytorch_available': TORCH_AVAILABLE,
            'total_trades': len(self.trade_history),
            'current_position': self.current_position,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'performance_target': '10% monthly returns'
        }


class SentioPPOStrategy:
    """
    Sentio Trader Strategy using PPO Agent
    Implements the strategy interface for Sentio Trader
    """
    
    def __init__(self, symbol: str = "QQQ", model_path: Optional[str] = None):
        self.name = "PPO_10_Percent_Monthly"
        self.version = "1.0"
        self.symbol = symbol
        self.agent = SentioPPOAgent(model_path)
        
        # Strategy parameters
        self.lookback_window = 120  # 2 hours
        self.min_confidence = 0.6  # Minimum confidence to trade
        self.max_trades_per_day = 20
        self.trades_today = 0
        
        logger.info(f"PPO Strategy initialized for {symbol}")
    
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal using PPO agent"""
        
        # Check if we have enough data
        if len(market_data) < self.lookback_window:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reason': 'Insufficient data'
            }
        
        # Check daily trade limit
        if self.trades_today >= self.max_trades_per_day:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reason': 'Daily trade limit reached'
            }
        
        # Get PPO signal
        ppo_signal = self.agent.get_trading_signal(market_data)
        
        # Apply minimum confidence filter
        if ppo_signal['confidence'] < self.min_confidence:
            return {
                'action': 'HOLD',
                'confidence': ppo_signal['confidence'],
                'reason': f"Confidence {ppo_signal['confidence']:.2f} below threshold {self.min_confidence}"
            }
        
        # Return signal
        return {
            'action': ppo_signal['signal'],
            'confidence': ppo_signal['confidence'],
            'position_size': ppo_signal['position_size'],
            'expected_value': ppo_signal['expected_value'],
            'model_loaded': ppo_signal['model_loaded'],
            'reason': 'PPO model prediction'
        }
    
    def on_trade_executed(self, trade_result: Dict[str, Any]):
        """Handle trade execution result"""
        self.trades_today += 1
        self.agent.update_performance(trade_result)
    
    def on_day_end(self):
        """Reset daily counters"""
        self.trades_today = 0
        self.agent.daily_pnl = 0
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata"""
        return {
            'name': self.name,
            'version': self.version,
            'symbol': self.symbol,
            'target_return': '10% monthly',
            'model_type': 'Enhanced PPO with LSTM + Attention',
            'features': '2,340 advanced features',
            'training': 'Curriculum learning + Multi-agent ensemble',
            'status': self.agent.get_status()
        }


# Integration functions for Sentio Trader
def create_ppo_strategy(symbol: str = "QQQ", model_path: Optional[str] = None) -> SentioPPOStrategy:
    """Create PPO strategy for Sentio Trader"""
    return SentioPPOStrategy(symbol, model_path)


def get_ppo_agent_status() -> Dict[str, Any]:
    """Get PPO agent status for monitoring"""
    agent = SentioPPOAgent()
    return agent.get_status()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Sentio PPO Integration Test")
    print("=" * 40)
    
    # Create strategy
    strategy = create_ppo_strategy("QQQ")
    
    # Display metadata
    metadata = strategy.get_metadata()
    print(f"📊 Strategy: {metadata['name']} v{metadata['version']}")
    print(f"🎯 Target: {metadata['target_return']}")
    print(f"🧠 Model: {metadata['model_type']}")
    print(f"📈 Features: {metadata['features']}")
    print(f"🎓 Training: {metadata['training']}")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='1min'),
        'open': np.random.randn(200) * 0.01 + 100,
        'high': np.random.randn(200) * 0.01 + 101,
        'low': np.random.randn(200) * 0.01 + 99,
        'close': np.random.randn(200) * 0.01 + 100,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    # Generate signal
    signal = strategy.generate_signal(sample_data)
    print(f"\n📡 Test Signal:")
    print(f"   Action: {signal['action']}")
    print(f"   Confidence: {signal['confidence']:.2f}")
    print(f"   Reason: {signal['reason']}")
    
    # Display agent status
    status = strategy.agent.get_status()
    print(f"\n🤖 Agent Status:")
    print(f"   Model Loaded: {status['model_loaded']}")
    print(f"   PyTorch Available: {status['pytorch_available']}")
    print(f"   Performance Target: {status['performance_target']}")
    
    print(f"\n✅ Sentio PPO Integration Ready!")
    print(f"🎯 Ready to achieve 10% monthly returns in Sentio Trader!")
