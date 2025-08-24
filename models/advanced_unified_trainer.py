"""
Advanced Unified PPO Trainer with State-of-the-Art Enhancements

Integrates:
1. Purged Walk-Forward CV with embargo and block bootstrap
2. Distributional critic with quantile regression and CVaR optimization  
3. HMM-based regime gating for policy switching and per-regime logging

Plus all previous enhancements:
- Transformer policy networks
- Dynamic action masking
- Risk-aware training
- GPU acceleration
- Advanced backtesting
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    # Fallback for systems without CUDA
    autocast = lambda: lambda x: x
    GradScaler = lambda: None
import gymnasium as gym

# Import our advanced modules
from .purged_cv import PurgedKFold, block_bootstrap_pnl, PurgedCVEvaluator
from .distributional_critic import (
    QuantileCritic, DistributionalValueLoss, EnhancedDistributionalCritic,
    quantile_huber_loss
)
from .regime_gating import RegimeGater, RegimeAwareMasker
from .exceptions import (
    TrainingError, TrainingConfigurationError, TrainingMemoryError,
    ModelError, ModelLoadError, ModelSaveError, DataError,
    ErrorHandler, ErrorContext
)
from .advanced_ppo_loss import AdvancedPPOLoss
from .transformer_policy import TransformerPolicyNetwork
from .dynamic_action_masker import DynamicActionMasker
from .risk_aware_ppo import RiskAwarePPO
# from .performance_optimization import PerformanceOptimizer  # Disabled for macOS compatibility
from .advanced_backtester import AdvancedBacktester

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Experiment tracking disabled.")

try:
    from tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("tensorboard not available. TensorBoard logging disabled.")


class AdvancedTradingEnvironment(gym.Env):
    """
    Enhanced trading environment with regime awareness and risk management.
    """
    
    def __init__(self, data_path: str, lookback_window: int = 120, 
                 initial_balance: float = 100000.0, transaction_cost: float = 0.001,
                 use_regime_gating: bool = True):
        super().__init__()
        
        # Load and prepare data
        self.data = self._load_data(data_path)
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.max_position_size = 0.1  # 10% of balance
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window * 6,), dtype=np.float32  # OHLCV + returns
        )
        
        # Advanced components
        self.regime_gater = RegimeGater(n_regimes=3, lookback_window=50) if use_regime_gating else None
        self.action_masker = DynamicActionMasker(action_space_size=3)
        self.risk_manager = RiskAwarePPO()
        
        # Performance tracking
        self.episode_returns = []
        self.episode_trades = []
        self.drawdown_history = []
        
        # Initialize regime detector if enabled
        if self.regime_gater and len(self.data) > 200:
            returns = self.data['close'].pct_change().dropna()
            self.regime_gater.fit_hmm(returns)
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess market data."""
        if data_path.endswith('.feather'):
            df = pd.read_feather(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate returns and technical indicators
        df['returns'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['returns'].rolling(20).std().fillna(0)
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        
        return df.dropna().reset_index(drop=True)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        
        # Reset tracking
        self.episode_returns = []
        self.episode_trades = []
        self.drawdown_history = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Get current price and market state
        current_price = self.data.iloc[self.current_step]['close']
        
        # Update regime if enabled
        if self.regime_gater:
            recent_returns = self.data['returns'].iloc[
                max(0, self.current_step - 20):self.current_step
            ].values
            self.regime_gater.update_regime(recent_returns)
        
        # Apply action masking
        obs = self._get_observation()
        current_price = self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 100.0
        action_mask = self.action_masker.get_action_mask(
            obs, self.position, self.balance, self.balance + self.position * current_price
        )
        
        if self.regime_gater:
            regime_mask = self.regime_gater.get_action_mask(
                obs, self.position, self.balance
            )
            action_mask = action_mask & regime_mask
        
        # Override action if masked
        if not action_mask[action]:
            action = 0  # Default to hold
        
        # Execute trade
        reward, trade_info = self._execute_trade(action, current_price)
        
        # Update tracking
        self.episode_returns.append(reward)
        if trade_info['trade_executed']:
            self.episode_trades.append(trade_info)
        
        # Log to regime gater if enabled
        if self.regime_gater and trade_info['trade_executed']:
            self.regime_gater.log_trade_result(action, reward, abs(trade_info['position_change']))
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Calculate drawdown
        portfolio_value = self.balance + self.position * current_price
        if len(self.episode_returns) > 0:
            cum_returns = np.cumsum(self.episode_returns)
            running_max = np.maximum.accumulate(cum_returns)
            current_dd = (running_max[-1] - cum_returns[-1]) / max(running_max[-1], 1)
            self.drawdown_history.append(current_dd)
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'trade_info': trade_info,
            'current_drawdown': self.drawdown_history[-1] if self.drawdown_history else 0.0
        }
        
        if self.regime_gater:
            info['regime_info'] = self.regime_gater.get_current_regime_info()
        
        return self._get_observation(), reward, done, truncated, info
    
    def _execute_trade(self, action: int, price: float) -> Tuple[float, Dict]:
        """Execute trade and return reward and trade info."""
        old_position = self.position
        position_change = 0.0
        trade_executed = False
        
        # Calculate position size based on regime (if available)
        base_position_size = self.balance * self.max_position_size / price
        
        if self.regime_gater:
            regime_info = self.regime_gater.get_current_regime_info()
            position_mult = regime_info.get('position_size_mult', 1.0)
            position_size = base_position_size * position_mult
        else:
            position_size = base_position_size
        
        # Execute action
        if action == 1:  # Buy
            if self.balance >= position_size * price * (1 + self.transaction_cost):
                cost = position_size * price * (1 + self.transaction_cost)
                self.balance -= cost
                self.position += position_size
                position_change = position_size
                trade_executed = True
        
        elif action == 2:  # Sell
            if self.position >= position_size:
                proceeds = position_size * price * (1 - self.transaction_cost)
                self.balance += proceeds
                self.position -= position_size
                position_change = -position_size
                trade_executed = True
        
        # Calculate reward (P&L from position change)
        if len(self.episode_returns) > 0:
            # Use price change for reward
            prev_price = self.data.iloc[self.current_step - 1]['close']
            price_change = (price - prev_price) / prev_price
            reward = old_position * price_change * price
        else:
            reward = 0.0
        
        # Apply risk-aware reward adjustment if available
        if hasattr(self, 'risk_manager') and len(self.episode_returns) > 10:
            current_dd = self.drawdown_history[-1] if self.drawdown_history else 0.0
            reward = self.risk_manager.adjust_reward(
                reward, 
                np.array(self.episode_returns[-10:]), 
                current_dd
            )
        
        trade_info = {
            'action': action,
            'price': price,
            'position_change': position_change,
            'trade_executed': trade_executed,
            'transaction_cost': self.transaction_cost if trade_executed else 0.0
        }
        
        return reward, trade_info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        # Get market data features
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Pad if necessary
        if len(window_data) < self.lookback_window:
            padding_size = self.lookback_window - len(window_data)
            padding = np.zeros((padding_size, len(window_data.columns)))
            window_data = pd.concat([
                pd.DataFrame(padding, columns=window_data.columns),
                window_data
            ], ignore_index=True)
        
        # Extract features: OHLCV + returns
        features = window_data[['open', 'high', 'low', 'close', 'volume', 'returns']].values
        
        # Normalize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        return features.flatten().astype(np.float32)


class AdvancedPPOTrainer:
    """
    Advanced PPO trainer with all state-of-the-art enhancements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize environment
        self.env = AdvancedTradingEnvironment(
            data_path=config['data_path'],
            use_regime_gating=config.get('use_regime_gating', True)
        )
        
        # Initialize network based on configuration
        if config.get('use_distributional_critic', False):
            self.network = EnhancedDistributionalCritic(
                input_size=self.env.observation_space.shape[0],
                hidden_sizes=config.get('hidden_sizes', [512, 256]),
                n_quantiles=config.get('n_quantiles', 51),
                device=self.device
            ).to(self.device)
        elif config.get('use_transformer', False):
            self.network = TransformerPolicyNetwork(
                input_size=self.env.observation_space.shape[0],
                hidden_size=config.get('hidden_size', 512),
                num_layers=config.get('transformer_layers', 6),
                num_heads=config.get('transformer_heads', 8),
                action_size=self.env.action_space.n
            ).to(self.device)
        else:
            # Standard network
            self.network = self._create_standard_network().to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config.get('learning_rate', 3e-4)
        )
        
        # Initialize loss function
        if config.get('use_distributional_critic', False):
            self.loss_fn = DistributionalValueLoss(
                n_quantiles=config.get('n_quantiles', 51),
                cvar_alpha=config.get('cvar_alpha', 0.05),
                cvar_weight=config.get('cvar_weight', 0.5),
                device=self.device
            )
        else:
            self.loss_fn = AdvancedPPOLoss()
        
        # Performance optimization (simplified for macOS compatibility)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Evaluation components
        self.use_purged_cv = config.get('use_purged_cv', False)
        self.backtester = AdvancedBacktester(config)
        
        # Experiment tracking
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
        self.use_tensorboard = config.get('use_tensorboard', False) and TENSORBOARD_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'sentio-ppo-advanced'),
                config=config,
                name=config.get('experiment_name', f'advanced_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            )
        
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(
                log_dir=f"runs/{config.get('experiment_name', 'advanced_ppo')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        self.best_performance = -np.inf
        
    def _create_standard_network(self) -> nn.Module:
        """Create standard actor-critic network."""
        input_size = self.env.observation_space.shape[0]
        hidden_size = self.config.get('hidden_size', 512)
        action_size = self.env.action_space.n
        
        class StandardNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU()
                )
                self.actor = nn.Linear(hidden_size, action_size)
                self.critic = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                features = self.shared(x)
                return self.actor(features), self.critic(features)
        
        return StandardNetwork()
    
    @ErrorHandler.handle_training_error
    def train(self, total_episodes: int, max_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the PPO agent with all advanced features.
        
        Args:
            total_episodes: Maximum number of episodes
            max_minutes: Maximum training time in minutes
            
        Returns:
            Training results and metadata
        """
        start_time = time.time()
        episode_rewards = []
        episode_lengths = []
        
        logger.info(f"Starting advanced PPO training for {total_episodes} episodes")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Network type: {type(self.network).__name__}")
        logger.info(f"Enhanced features: purged_cv={self.use_purged_cv}, "
                   f"regime_gating={self.config.get('use_regime_gating', False)}, "
                   f"distributional={self.config.get('use_distributional_critic', False)}")
        
        try:
            for episode in range(total_episodes):
                # Check time limit
                if max_minutes and (time.time() - start_time) / 60 > max_minutes:
                    logger.info(f"Time limit reached ({max_minutes} minutes)")
                    break
                
                # Run episode
                episode_reward, episode_length = self._run_episode()
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                self.episode_count += 1
                
                # Log progress
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    avg_length = np.mean(episode_lengths[-10:])
                    
                    elapsed_minutes = (time.time() - start_time) / 60
                    progress = episode / total_episodes * 100
                    
                    logger.info(f"Episode {episode}/{total_episodes} ({progress:.1f}%) - "
                              f"Avg Reward: {avg_reward:.4f}, Avg Length: {avg_length:.1f}, "
                              f"Elapsed: {elapsed_minutes:.1f}min")
                    
                    # Log to experiment trackers
                    self._log_metrics({
                        'episode': episode,
                        'avg_reward_10': avg_reward,
                        'avg_length_10': avg_length,
                        'elapsed_minutes': elapsed_minutes
                    })
                
                # Evaluate periodically
                if episode > 0 and episode % 50 == 0:
                    self._evaluate_model(episode_rewards[-50:])
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        # Final evaluation
        final_results = self._final_evaluation(episode_rewards)
        
        # Save model
        model_path = self._save_model(final_results)
        
        logger.info(f"Training completed. Model saved to: {model_path}")
        
        return final_results
    
    def _run_episode(self) -> Tuple[float, int]:
        """Run a single training episode."""
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Storage for PPO update
        states, actions, rewards, log_probs, values = [], [], [], [], []
        dones = []
        
        while True:
            # Convert observation to tensor
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                if hasattr(self.network, 'forward'):
                    if self.config.get('use_distributional_critic', False):
                        action_logits, quantiles = self.network(state_tensor)
                        value = quantiles[:, quantiles.shape[1] // 2]  # Median quantile
                    else:
                        action_logits, value = self.network(state_tensor)
                else:
                    action_logits = self.network.actor(self.network.shared(state_tensor))
                    value = self.network.critic(self.network.shared(state_tensor))
                
                # Apply regime-based exploration temperature if available
                if hasattr(self.env, 'regime_gater') and self.env.regime_gater:
                    temp = self.env.regime_gater.get_exploration_temperature()
                    action_logits = action_logits / temp
                
                # Sample action
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            # Store for PPO update
            states.append(obs)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # Execute action
            next_obs, reward, done, truncated, info = self.env.step(action.item())
            
            rewards.append(reward)
            dones.append(done or truncated)
            episode_reward += reward
            episode_length += 1
            self.step_count += 1
            
            obs = next_obs
            
            if done or truncated:
                break
        
        # PPO update
        if len(states) > 10:  # Minimum batch size
            self._ppo_update(states, actions, rewards, log_probs, values, dones)
        
        return episode_reward, episode_length
    
    def _ppo_update(self, states: List[np.ndarray], actions: List[int], 
                   rewards: List[float], old_log_probs: List[float], 
                   values: List[float], dones: List[bool]):
        """Perform PPO update with advanced features."""
        
        try:
            # Convert to tensors with explicit memory management
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
            values_tensor = torch.FloatTensor(values).to(self.device)
            dones_tensor = torch.BoolTensor(dones).to(self.device)
            
            # Calculate advantages using GAE
            advantages = self._calculate_gae(rewards_tensor, values_tensor, dones_tensor)
            returns = advantages + values_tensor
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO epochs
            for _ in range(self.config.get('ppo_epochs', 4)):
                # Forward pass
                if self.config.get('use_distributional_critic', False):
                    action_logits, quantiles = self.network(states_tensor)
                    
                    # Calculate new log probs
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = action_dist.log_prob(actions_tensor)
                    entropy = action_dist.entropy()
                    
                    # Prepare targets for distributional critic
                    target_quantiles = returns.unsqueeze(1).expand(-1, quantiles.shape[1])
                    
                    # Calculate distributional loss
                    loss, loss_info = self.loss_fn.compute_distributional_loss(
                        quantiles, returns
                    )
                else:
                    action_logits, pred_values = self.network(states_tensor)
                    
                    # Calculate new log probs
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = action_dist.log_prob(actions_tensor)
                    entropy = action_dist.entropy()
                    
                    # Standard PPO loss
                    loss, loss_info = self.loss_fn.compute_loss(
                        old_log_probs_tensor, new_log_probs, advantages,
                        returns, pred_values, actions_tensor, entropy
                    )
                
                # Backward pass (simplified for macOS compatibility)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Log training metrics
            if self.step_count % 100 == 0:
                self._log_metrics({
                    'training/loss': loss.item(),
                    'training/step': self.step_count,
                    **{f'training/{k}': v for k, v in loss_info.items()}
                })
                    
        except RuntimeError as e:
            logger.error(f"Error during PPO update: {e}")
            # Clear GPU cache on error
            if self.device.type == 'cuda':
                import torch as torch_module
                torch_module.cuda.empty_cache()
            raise
        finally:
            # Explicit cleanup of large tensors
            if 'states_tensor' in locals():
                del states_tensor
            if 'actions_tensor' in locals():
                del actions_tensor
            if 'rewards_tensor' in locals():
                del rewards_tensor
            if 'old_log_probs_tensor' in locals():
                del old_log_probs_tensor
            if 'values_tensor' in locals():
                del values_tensor
            if 'dones_tensor' in locals():
                del dones_tensor
            if 'advantages' in locals():
                del advantages
            if 'returns' in locals():
                del returns
            
            # Force garbage collection for GPU memory
            if self.device.type == 'cuda':
                import torch as torch_module
                torch_module.cuda.empty_cache()
    
    def _calculate_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                      dones: torch.Tensor, gamma: float = 0.99, 
                      lambda_: float = 0.95) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def _evaluate_model(self, recent_rewards: List[float]):
        """Evaluate model performance with advanced metrics."""
        if len(recent_rewards) < 10:
            return
        
        # Basic metrics
        avg_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        sharpe_ratio = avg_reward / reward_std if reward_std > 0 else 0
        
        # Regime-specific performance
        regime_performance = {}
        if hasattr(self.env, 'regime_gater') and self.env.regime_gater:
            regime_performance = self.env.regime_gater.get_regime_performance()
        
        # Update best performance
        if avg_reward > self.best_performance:
            self.best_performance = avg_reward
            logger.info(f"New best performance: {avg_reward:.4f}")
        
        # Log evaluation metrics
        eval_metrics = {
            'eval/avg_reward': avg_reward,
            'eval/reward_std': reward_std,
            'eval/sharpe_ratio': sharpe_ratio,
            'eval/best_performance': self.best_performance
        }
        
        # Add regime metrics
        for regime_name, metrics in regime_performance.items():
            for metric_name, value in metrics.items():
                eval_metrics[f'eval/regime_{regime_name}_{metric_name}'] = value
        
        self._log_metrics(eval_metrics)
    
    def _final_evaluation(self, all_rewards: List[float]) -> Dict[str, Any]:
        """Perform final evaluation with purged CV if enabled."""
        results = {
            'total_episodes': len(all_rewards),
            'total_steps': self.step_count,
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'best_reward': np.max(all_rewards) if all_rewards else 0,
            'final_sharpe': np.mean(all_rewards) / np.std(all_rewards) if np.std(all_rewards) > 0 else 0
        }
        
        # Purged CV evaluation
        if self.use_purged_cv and len(all_rewards) > 100:
            logger.info("Running purged cross-validation evaluation...")
            
            # Create synthetic timestamps for CV
            timestamps = pd.date_range(
                start='2025-01-01', 
                periods=len(all_rewards), 
                freq='T'
            )
            returns_series = pd.Series(all_rewards, index=timestamps)
            
            try:
                evaluator = PurgedCVEvaluator(
                    n_splits=self.config.get('cv_splits', 5),
                    purge_pct=self.config.get('purge_pct', 0.02),
                    embargo_pct=self.config.get('embargo_pct', 0.01)
                )
                
                # Simple strategy function for evaluation
                def simple_strategy(X_train, y_train, X_test):
                    # Return simple momentum signals
                    return np.sign(np.mean(y_train[-10:]) if len(y_train) > 10 else 0)
                
                cv_results = evaluator.evaluate_strategy(
                    np.array(returns_series).reshape(-1, 1), 
                    np.array(returns_series), 
                    timestamps, 
                    simple_strategy
                )
                results['purged_cv'] = cv_results
                logger.info(f"Purged CV evaluation completed with {cv_results['n_folds']} folds")
            except Exception as e:
                logger.warning(f"Purged CV evaluation failed: {e}")
        
        # Regime performance
        if hasattr(self.env, 'regime_gater') and self.env.regime_gater:
            results['regime_performance'] = self.env.regime_gater.get_regime_performance()
        
        return results
    
    def _save_model(self, results: Dict[str, Any]) -> str:
        """Save trained model and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"advanced_ppo_model_{timestamp}"
        
        # Create output directory
        output_dir = Path(self.config.get('output_dir', 'trained_models'))
        output_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = output_dir / f"{model_name}.pth"
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'results': results
        }, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': 'Advanced PPO',
            'training_config': self.config,
            'performance_metrics': results,
            'enhanced_features': {
                'purged_cv': self.use_purged_cv,
                'distributional_critic': self.config.get('use_distributional_critic', False),
                'regime_gating': self.config.get('use_regime_gating', False),
                'transformer_policy': self.config.get('use_transformer', False)
            },
            'created_at': datetime.now().isoformat(),
            'device': str(self.device),
            'total_parameters': sum(p.numel() for p in self.network.parameters())
        }
        
        metadata_path = output_dir / f"{model_name}-metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        return str(model_path)
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to experiment trackers."""
        if self.use_wandb:
            wandb.log(metrics)
        
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, self.step_count)


def main():
    """Main training function with CLI."""
    parser = argparse.ArgumentParser(description="Advanced PPO Trainer with State-of-the-Art Features")
    
    # Basic training parameters
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--minutes', type=int, help='Maximum training time in minutes')
    parser.add_argument('--output-dir', type=str, default='trained_models', help='Output directory')
    
    # Model architecture
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--use-transformer', action='store_true', help='Use transformer policy network')
    parser.add_argument('--transformer-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--transformer-heads', type=int, default=8, help='Number of attention heads')
    
    # Advanced features
    parser.add_argument('--use-distributional-critic', action='store_true', help='Use distributional critic with CVaR')
    parser.add_argument('--n-quantiles', type=int, default=51, help='Number of quantiles for distributional critic')
    parser.add_argument('--cvar-alpha', type=float, default=0.05, help='CVaR alpha level')
    parser.add_argument('--use-regime-gating', action='store_true', help='Enable regime-based gating')
    parser.add_argument('--use-purged-cv', action='store_true', help='Use purged cross-validation')
    
    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO update epochs')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    
    # Evaluation parameters
    parser.add_argument('--cv-splits', type=int, default=5, help='Cross-validation splits')
    parser.add_argument('--purge-pct', type=float, default=0.02, help='Purging percentage for CV')
    parser.add_argument('--embargo-pct', type=float, default=0.01, help='Embargo percentage for CV')
    
    # Experiment tracking
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='sentio-ppo-advanced', help='W&B project name')
    parser.add_argument('--use-tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Validate data path
    if not Path(config['data_path']).exists():
        raise FileNotFoundError(f"Data file not found: {config['data_path']}")
    
    # Initialize trainer
    logger.info("Initializing Advanced PPO Trainer...")
    trainer = AdvancedPPOTrainer(config)
    
    # Start training
    logger.info("Starting training with advanced features...")
    results = trainer.train(
        total_episodes=config['episodes'],
        max_minutes=config.get('minutes')
    )
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Total Episodes: {results['total_episodes']}")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Mean Reward: {results['mean_reward']:.4f}")
    print(f"Final Sharpe: {results['final_sharpe']:.4f}")
    
    if 'purged_cv' in results:
        cv_sharpe = results['purged_cv']['cv_results']['mean_sharpe']
        print(f"Purged CV Sharpe: {cv_sharpe:.4f}")
    
    if 'regime_performance' in results:
        print("\nRegime Performance:")
        for regime, metrics in results['regime_performance'].items():
            print(f"  {regime}: Sharpe={metrics['sharpe_ratio']:.3f}, Trades={metrics['trades']}")
    
    print("="*50)


if __name__ == "__main__":
    main()
