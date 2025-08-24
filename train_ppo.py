#!/usr/bin/env python3
"""
Fixed PPO Training Interface - All Critical Bugs Resolved
Fixes reward scaling, trading incentives, projection calculations, and portfolio tracking.
"""

import argparse
import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

def setup_logging(ppo_type: str, output_name: str = None) -> str:
    """Setup comprehensive logging to file for debugging."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ppo_training_{ppo_type}_{output_name or 'default'}_{timestamp}.log"
    log_path = logs_dir / log_filename
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 PPO Training Session Started (FIXED VERSION)")
    logger.info(f"📝 Log file: {log_path}")
    logger.info(f"🤖 PPO Type: {ppo_type}")
    logger.info(f"💾 Output: {output_name or 'default'}")
    logger.info(f"⏰ Session: {timestamp}")
    logger.info("=" * 80)
    
    return str(log_path)

logger = logging.getLogger(__name__)

class TradingEnvironment:
    """Fixed trading environment with proper reward structure and trading incentives."""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, 
                 max_episode_steps: int = 1000, random_start: bool = True):
        self.data = data
        self.initial_balance = initial_balance
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        
        # FIX: Track peak portfolio value for drawdown calculation
        self.peak_portfolio_value = initial_balance
        
        logger.info(f"🏗️ Environment initialized (FIXED):")
        logger.info(f"   Data length: {len(data):,} rows")
        logger.info(f"   Max episode steps: {max_episode_steps:,}")
        logger.info(f"   Episode length: ~{max_episode_steps/390:.1f} trading days")
        logger.info(f"   Random start: {random_start}")
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state."""
        if self.random_start and len(self.data) > self.max_episode_steps + 100:
            max_start = len(self.data) - self.max_episode_steps - 50
            self.episode_start_idx = np.random.randint(50, max_start)
        else:
            self.episode_start_idx = 50
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_trades = 0
        self.episode_reward = 0.0
        self.trade_history = []
        
        # FIX: Reset portfolio tracking
        self.last_portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.consecutive_holds = 0  # Track consecutive hold actions
        
        logger.debug(f"🔄 Episode reset - Start idx: {self.episode_start_idx}")
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current market observation with 20 features."""
        data_idx = self.episode_start_idx + self.current_step
        
        if data_idx >= len(self.data) - 1:
            return np.zeros(20, dtype=np.float32)
        
        lookback = min(20, data_idx)
        
        # Price features
        current_price = self.data.iloc[data_idx]['close']
        if lookback > 0:
            recent_prices = self.data.iloc[data_idx-lookback:data_idx]['close'].values
            price_change = (current_price - recent_prices[-1]) / recent_prices[-1] if len(recent_prices) > 0 else 0.0
            volatility = np.std(recent_prices) / np.mean(recent_prices) if len(recent_prices) > 1 else 0.0
        else:
            price_change = 0.0
            volatility = 0.0
        
        # Volume features
        current_volume = self.data.iloc[data_idx]['volume']
        if lookback > 0:
            recent_volumes = self.data.iloc[data_idx-lookback:data_idx]['volume'].values
            volume_ratio = current_volume / np.mean(recent_volumes) if len(recent_volumes) > 0 and np.mean(recent_volumes) > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        # Technical indicators
        if lookback >= 10:
            prices = self.data.iloc[data_idx-lookback:data_idx]['close'].values
            sma_short = np.mean(prices[-5:]) if len(prices) >= 5 else current_price
            sma_long = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
            rsi = self._calculate_rsi(prices) if len(prices) >= 14 else 50.0
        else:
            sma_short = sma_long = current_price
            rsi = 50.0
        
        # Portfolio features
        portfolio_value = self.balance + self.position * current_price
        position_ratio = self.position * current_price / portfolio_value if portfolio_value > 0 else 0.0
        
        # Risk features
        drawdown = max(0, (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value)
        
        # FIX: Add trade frequency feature to encourage trading
        trade_frequency = self.total_trades / max(1, self.current_step) if self.current_step > 0 else 0.0
        
        # Create 20-feature observation
        obs = np.array([
            # Price features (6)
            current_price / 100.0,
            price_change,
            volatility,
            (sma_short / current_price) - 1.0 if current_price > 0 else 0.0,
            (sma_long / current_price) - 1.0 if current_price > 0 else 0.0,
            (rsi / 100.0) - 0.5,
            
            # Volume features (2)
            np.log(max(1, current_volume)) / 15.0,
            np.clip(volume_ratio - 1.0, -2.0, 2.0),
            
            # Portfolio features (4)
            (self.balance / self.initial_balance) - 1.0,
            np.clip(position_ratio, -1.0, 1.0),
            (portfolio_value / self.initial_balance) - 1.0,
            drawdown,
            
            # Market timing (4)
            self.current_step / max(1, self.max_episode_steps),
            (self.current_step % 390) / 390.0,
            np.sin(2 * np.pi * self.current_step / 390),
            np.cos(2 * np.pi * self.current_step / 390),
            
            # Trading features (4)
            trade_frequency,  # FIX: Use actual trade frequency
            1.0 if self.position > 0 else 0.0,
            1.0 if self.position < 0 else 0.0,
            min(1.0, self.consecutive_holds / 10.0)  # FIX: Add consecutive holds feature
        ], dtype=np.float32)
        
        # Ensure no NaN values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute trading action with fixed reward structure."""
        data_idx = self.episode_start_idx + self.current_step
        
        # Check if episode should end
        if (self.current_step >= self.max_episode_steps or 
            data_idx >= len(self.data) - 1):
            # FIX: Add final episode statistics
            portfolio_value = self.balance + (self.position * self.data.iloc[data_idx]['close'] if self.position > 0 else 0)
            total_return = (portfolio_value - self.initial_balance) / self.initial_balance
            
            logger.info(f"📊 Episode End: Portfolio ${portfolio_value:.2f} | "
                       f"Return: {total_return:.2%} | Trades: {self.total_trades} | "
                       f"Avg Reward: {self.episode_reward/max(1, self.current_step):.6f}")
            
            return self._get_observation(), 0.0, True, False, {
                'reason': 'max_steps',
                'portfolio_value': portfolio_value,
                'total_return': total_return,
                'total_trades': self.total_trades
            }
        
        current_price = self.data.iloc[data_idx]['close']
        old_portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)
        
        reward = 0.0
        transaction_cost = 0.001  # 0.1% transaction cost
        action_taken = False
        
        # Execute action
        if action == 1 and self.balance > current_price * (1 + transaction_cost):
            # Buy action
            shares_to_buy = self.balance / (current_price * (1 + transaction_cost))
            cost = shares_to_buy * current_price * (1 + transaction_cost)
            self.balance -= cost
            self.position += shares_to_buy
            self.total_trades += 1
            action_taken = True
            self.consecutive_holds = 0
            
            self.trade_history.append({
                'action': 'buy',
                'price': current_price,
                'shares': shares_to_buy,
                'step': self.current_step,
                'profit': 0.0
            })
            
        elif action == 2 and self.position > 0:
            # Sell action
            proceeds = self.position * current_price * (1 - transaction_cost)
            # Calculate profit from average buy price
            avg_buy_price = sum([t['price'] * t['shares'] for t in self.trade_history if t['action'] == 'buy']) / max(1, self.position)
            profit = proceeds - (self.position * avg_buy_price)
            
            self.balance += proceeds
            self.position = 0
            self.total_trades += 1
            action_taken = True
            self.consecutive_holds = 0
            
            if self.trade_history:
                self.trade_history[-1]['profit'] = profit
        else:
            # Hold action
            self.consecutive_holds += 1
        
        # Move to next step
        self.current_step += 1
        next_data_idx = self.episode_start_idx + self.current_step
        
        if next_data_idx < len(self.data):
            next_price = self.data.iloc[next_data_idx]['close']
        else:
            next_price = current_price
        
        # Calculate new portfolio value
        new_portfolio_value = self.balance + (self.position * next_price if self.position > 0 else 0)
        
        # Update peak portfolio value
        if new_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_portfolio_value
        
        # FIX: Improved reward calculation
        # 1. Base reward: portfolio change (scaled appropriately)
        pct_change = (new_portfolio_value - old_portfolio_value) / old_portfolio_value if old_portfolio_value > 0 else 0.0
        base_reward = pct_change * 100  # FIX: Scale by 100 instead of 1000 (more stable)
        
        # 2. Trading encouragement bonus (to prevent HOLD-only strategy)
        trading_bonus = 0.0
        if action_taken:
            trading_bonus = 0.01  # Small bonus for taking action
            
            # Extra bonus for profitable trades
            if len(self.trade_history) > 0 and self.trade_history[-1].get('profit', 0) > 0:
                trading_bonus += 0.02
        
        # 3. Penalty for excessive holding (to encourage exploration)
        holding_penalty = 0.0
        if self.consecutive_holds > 50:  # Penalize after 50 consecutive holds
            holding_penalty = 0.001 * (self.consecutive_holds - 50)
        
        # 4. Risk management bonus/penalty
        risk_reward = 0.0
        current_drawdown = (self.peak_portfolio_value - new_portfolio_value) / self.peak_portfolio_value
        if current_drawdown > 0.1:  # Penalty for >10% drawdown
            risk_reward = -0.05
        elif current_drawdown < 0.02 and new_portfolio_value > self.initial_balance:  # Bonus for low drawdown with profit
            risk_reward = 0.01
        
        # Combine all reward components
        reward = base_reward + trading_bonus - holding_penalty + risk_reward
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -10.0, 10.0)
        
        # Update tracking
        self.last_portfolio_value = new_portfolio_value
        self.episode_reward += reward
        
        # Log detailed info periodically
        if self.current_step % 100 == 0:
            logger.debug(f"Step {self.current_step}: Action {action} | Portfolio ${new_portfolio_value:.2f} | "
                       f"Reward: {reward:.4f} (base:{base_reward:.4f}, bonus:{trading_bonus:.4f}, "
                       f"hold_pen:{holding_penalty:.4f}, risk:{risk_reward:.4f}) | "
                       f"Trades: {self.total_trades} | ConsecHolds: {self.consecutive_holds}")
        
        # Check if done
        done = (self.current_step >= self.max_episode_steps or 
                next_data_idx >= len(self.data) - 1)
        
        return self._get_observation(), reward, done, False, {
            'portfolio_value': new_portfolio_value,
            'total_trades': self.total_trades,
            'episode_reward': self.episode_reward,
            'consecutive_holds': self.consecutive_holds
        }

class PPONetwork(nn.Module):
    """PPO Network with improved initialization and architecture."""
    
    def __init__(self, input_size=20, hidden_size=256, action_size=3, ppo_type='enhanced-maskable'):
        super().__init__()
        self.ppo_type = ppo_type
        self.action_size = action_size
        
        # Feature extraction with better initialization
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),  # FIX: Add LayerNorm for stability
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # FIX: Add LayerNorm
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        
        # Policy head with better initialization
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # FIX: Initialize weights properly to encourage exploration
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to encourage initial exploration."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # FIX: Initialize policy head with smaller weights for more exploration
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
    
    def forward(self, x, action_mask=None):
        features = self.feature_net(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        
        # Apply action masking for maskable variants
        if ('maskable' in self.ppo_type or 'enhanced' in self.ppo_type) and action_mask is not None:
            # FIX: Ensure at least one action is available
            if action_mask.sum() == 0:
                action_mask[0] = 1  # Allow HOLD if nothing else available
            logits = logits + (action_mask - 1) * 1e8
        
        return logits, value
    
    def get_action_mask(self, obs, position, balance, current_price):
        """Get action mask based on current state."""
        mask = torch.ones(self.action_size)
        
        # Can't buy if insufficient balance
        if balance < current_price * 1.001:
            mask[1] = 0
        
        # Can't sell if no position
        if position <= 0:
            mask[2] = 0
        
        return mask

class PPOTrainer:
    """Fixed PPO Trainer with improved training dynamics."""
    
    def __init__(self, 
                 ppo_type: str = 'enhanced-maskable',
                 data_path: str = 'data/polygon_QQQ_1m.feather',
                 output_name: Optional[str] = None,
                 learning_rate: float = 3e-4,
                 episode_length: int = 1000):
        
        self.ppo_type = ppo_type
        self.episode_length = episode_length
        self.device = torch.device('cpu')
        
        # Load data
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} rows of market data")
        logger.info(f"Episode length: {episode_length} steps (~{episode_length/390:.1f} trading days)")
        
        # Create environment
        self.env = TradingEnvironment(self.data, max_episode_steps=episode_length, random_start=True)
        
        # Create network
        self.network = PPONetwork(ppo_type=ppo_type)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # FIX: Add learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=20
        )
        
        # PPO hyperparameters (adjusted for better exploration)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.02  # FIX: Increased from 0.01 for more exploration
        self.value_coef = 0.5
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades = []  # FIX: Track trades per episode
        self.training_start_time = None
        
        # Output naming
        if output_name:
            self.output_name = output_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_name = f"{ppo_type}_fixed_{timestamp}"
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load market data from file."""
        path = Path(data_path)
        if path.exists():
            df = pd.read_feather(path)
            # Use recent data for training
            return df.tail(20000).reset_index(drop=True)
        else:
            logger.warning(f"Data file {data_path} not found, generating synthetic data")
            # Generate realistic synthetic data
            n_points = 10000
            dates = pd.date_range('2023-01-01', periods=n_points, freq='1min')
            
            # Generate price with trend and volatility
            returns = np.random.normal(0.0001, 0.02, n_points)
            price = 100 * np.exp(np.cumsum(returns))
            
            return pd.DataFrame({
                'open': price + np.random.normal(0, 0.1, n_points),
                'high': price + np.abs(np.random.normal(0, 0.2, n_points)),
                'low': price - np.abs(np.random.normal(0, 0.2, n_points)),
                'close': price,
                'volume': np.random.lognormal(8, 1, n_points).astype(int)
            }, index=dates)
    
    def collect_rollout(self, num_steps: int = 2000):
        """Collect experience rollout with exploration bonus."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        obs = self.env.reset()
        
        for step in range(num_steps):
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action mask for maskable variants
            action_mask = None
            if 'maskable' in self.ppo_type or 'enhanced' in self.ppo_type:
                current_idx = self.env.episode_start_idx + self.env.current_step
                if current_idx < len(self.env.data):
                    current_price = self.env.data.iloc[current_idx]['close']
                    action_mask = self.network.get_action_mask(
                        obs, self.env.position, self.env.balance, current_price
                    ).unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                logits, value = self.network(state_tensor, action_mask)
                
                # FIX: Add temperature for exploration (higher early in training)
                temperature = max(0.5, 1.0 - len(self.episode_rewards) / 1000.0)
                logits = logits / temperature
                
                probs = torch.softmax(logits, dim=-1)
                
                # FIX: Add epsilon-greedy exploration
                if np.random.random() < max(0.1, 0.3 - len(self.episode_rewards) / 500.0):
                    # Random valid action
                    if action_mask is not None:
                        valid_actions = torch.where(action_mask[0] > 0)[0]
                        if len(valid_actions) > 0:
                            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
                        else:
                            action = 0  # Default to HOLD
                    else:
                        action = np.random.randint(3)
                else:
                    action = torch.multinomial(probs, 1).item()
                
                log_prob = torch.log(probs[0, action] + 1e-8)
            
            # Environment step
            next_obs, reward, done, _, info = self.env.step(action)
            
            # Store experience
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            obs = next_obs
            
            if done:
                self.episode_rewards.append(self.env.episode_reward)
                self.episode_lengths.append(self.env.current_step)
                self.episode_trades.append(info.get('total_trades', 0))
                obs = self.env.reset()
        
        return states, actions, rewards, values, log_probs, dones
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def ppo_update(self, states, actions, old_log_probs, advantages, returns):
        """Perform PPO update with improved stability."""
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO epochs
        for epoch in range(4):
            logits, values = self.network(states_tensor)
            probs = torch.softmax(logits, dim=-1)
            new_log_probs = torch.log(probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8)
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns_tensor)
            
            # Entropy loss (for exploration)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, 
              max_episodes: Optional[int] = None, 
              max_minutes: Optional[int] = None,
              target_reward: float = 0.1) -> Dict:  # FIX: More reasonable target
        """Train the PPO model with fixed convergence criteria."""
        
        logger.info(f"Starting {self.ppo_type} PPO training (FIXED VERSION)")
        logger.info(f"Limits: {max_episodes or 'unlimited'} episodes, {max_minutes or 'unlimited'} minutes")
        logger.info(f"Target reward: {target_reward:.6f} per step")
        
        # FIX: Calculate realistic monthly target
        daily_return = target_reward * 390 / 100  # Adjusted for 100x scaling
        monthly_return = (1 + daily_return) ** 22 - 1 if abs(daily_return) < 0.1 else 0.0
        logger.info(f"🎯 Target: {monthly_return:.1%} monthly (22 trading days)")
        logger.info(f"🎯 Daily target: {daily_return:.3%} per day")
        
        self.training_start_time = time.time()
        iteration = 0
        best_avg_reward = -float('inf')
        
        try:
            while True:
                iteration += 1
                
                # Check stopping conditions
                if max_episodes and len(self.episode_rewards) >= max_episodes:
                    logger.info(f"Reached episode limit: {max_episodes}")
                    break
                
                if max_minutes:
                    elapsed_minutes = (time.time() - self.training_start_time) / 60
                    if elapsed_minutes >= max_minutes:
                        logger.info(f"Reached time limit: {max_minutes} minutes")
                        break
                
                # Check if model is improving and trading
                if not max_episodes and not max_minutes and len(self.episode_rewards) >= 100:
                    recent_rewards = self.episode_rewards[-50:]
                    recent_trades = self.episode_trades[-50:] if len(self.episode_trades) >= 50 else self.episode_trades
                    avg_reward = np.mean(recent_rewards)
                    avg_trades = np.mean(recent_trades)
                    reward_std = np.std(recent_rewards)
                    
                    # FIX: Convergence criteria - must be trading AND profitable
                    if avg_reward >= target_reward and avg_trades >= 5 and reward_std < 0.5:
                        logger.info(f"✅ Model converged! Avg reward: {avg_reward:.6f}, "
                                  f"Avg trades: {avg_trades:.1f}, Std: {reward_std:.6f}")
                        break
                    
                    # FIX: Stop if not improving after many episodes
                    if len(self.episode_rewards) > 500 and avg_trades < 1:
                        logger.warning(f"⚠️ Model not learning to trade after 500 episodes. "
                                     f"Avg trades: {avg_trades:.1f}")
                        break
                
                # Collect experience
                states, actions, rewards, values, log_probs, dones = self.collect_rollout(2000)
                
                # Compute advantages
                advantages, returns = self.compute_gae(rewards, values, dones)
                
                # PPO update
                loss_info = self.ppo_update(states, actions, log_probs, advantages, returns)
                
                # Update learning rate based on performance
                if len(self.episode_rewards) >= 10:
                    recent_avg = np.mean(self.episode_rewards[-10:])
                    self.scheduler.step(recent_avg)
                    if recent_avg > best_avg_reward:
                        best_avg_reward = recent_avg
                
                # Log progress every 5 iterations
                if iteration % 5 == 0:
                    recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                    recent_trades = self.episode_trades[-10:] if len(self.episode_trades) >= 10 else self.episode_trades
                    avg_reward = np.mean(recent_rewards)
                    avg_trades = np.mean(recent_trades) if recent_trades else 0
                    avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else 0
                    elapsed_time = (time.time() - self.training_start_time) / 60
                    
                    # FIX: Calculate realistic projected returns
                    if abs(avg_reward) > 0.01:  # Only project if reward is significant
                        projected_daily = max(-0.1, min(0.1, avg_reward * 390 / 100))  # Cap at ±10% daily
                        projected_monthly = (1 + projected_daily) ** 22 - 1
                    else:
                        projected_monthly = 0.0
                    
                    # Enhanced logging with trade statistics
                    reward_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0
                    min_reward = np.min(recent_rewards) if recent_rewards else 0
                    max_reward = np.max(recent_rewards) if recent_rewards else 0
                    
                    logger.info(f"Iteration {iteration} | Episodes: {len(self.episode_rewards)} | "
                              f"Avg Reward: {avg_reward:.6f} | Std: {reward_std:.6f} | "
                              f"Range: [{min_reward:.6f}, {max_reward:.6f}] | "
                              f"Avg Trades: {avg_trades:.1f} | Avg Length: {avg_length:.1f} | "
                              f"Time: {elapsed_time:.1f}min")
                    logger.info(f"📊 Loss Components: Policy={loss_info['policy_loss']:.4f}, "
                              f"Value={loss_info['value_loss']:.4f}, "
                              f"Entropy={loss_info['entropy']:.4f}, "
                              f"Total={loss_info['total_loss']:.4f}")
                    logger.info(f"📊 Projected: {projected_monthly:.1%} monthly (22 trading days)")
                    
                    # Log environment statistics every 50 iterations
                    if iteration % 50 == 0:
                        logger.info(f"🏗️ Environment Stats: Recent avg trades per episode: {avg_trades:.1f}")
                        logger.info(f"🏗️ Training Progress: Best avg reward so far: {best_avg_reward:.6f}")
            
            # Save model
            self._save_model()
            
            # Return results
            training_time = time.time() - self.training_start_time
            final_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0
            final_trades = np.mean(self.episode_trades[-10:]) if len(self.episode_trades) >= 10 else 0
            
            return {
                'success': True,
                'total_episodes': len(self.episode_rewards),
                'training_time_minutes': training_time / 60,
                'final_avg_reward': final_reward,
                'final_avg_trades': final_trades,
                'model_path': f"{self.output_name}.pth",
                'ppo_type': self.ppo_type
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                'success': False,
                'reason': str(e),
                'total_episodes': len(self.episode_rewards),
                'training_time_minutes': (time.time() - self.training_start_time) / 60 if self.training_start_time else 0
            }
    
    def _save_model(self):
        """Save trained model and metadata."""
        model_path = f"{self.output_name}.pth"
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ppo_type': self.ppo_type,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_trades': self.episode_trades
        }, model_path)
        
        # Save metadata
        metadata_path = f"{self.output_name}_metadata.json"
        metadata = {
            'ppo_type': self.ppo_type,
            'total_episodes': len(self.episode_rewards),
            'training_time_minutes': (time.time() - self.training_start_time) / 60 if self.training_start_time else 0,
            'final_avg_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
            'final_avg_trades': np.mean(self.episode_trades[-10:]) if len(self.episode_trades) >= 10 else 0,
            'model_path': model_path,
            'created_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Fixed PPO Training Interface - All Critical Bugs Resolved',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ppo_fixed.py                                    # Enhanced-maskable PPO, unlimited training
  python train_ppo_fixed.py --minutes 30                      # 30-minute training
  python train_ppo_fixed.py --episodes 100                    # 100 episodes
  python train_ppo_fixed.py --type standard                   # Standard PPO
  python train_ppo_fixed.py --type maskable                   # Maskable PPO
  python train_ppo_fixed.py --output fixed_model              # Custom output name
        """
    )
    
    parser.add_argument('--type', choices=['standard', 'maskable', 'enhanced-maskable'], 
                       default='enhanced-maskable', help='PPO variant to train (default: enhanced-maskable)')
    parser.add_argument('--minutes', type=int, help='Training time limit in minutes')
    parser.add_argument('--episodes', type=int, help='Maximum number of episodes')
    parser.add_argument('--output', type=str, help='Output filename (without extension)')
    parser.add_argument('--data', type=str, default='data/polygon_QQQ_1m.feather', 
                       help='Input data file path (default: data/polygon_QQQ_1m.feather)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--episode-length', type=int, default=1000, 
                       help='Steps per episode (default: 1000 = ~2.5 trading days)')
    
    args = parser.parse_args()
    
    # Setup comprehensive logging
    log_path = setup_logging(args.type, args.output)
    
    # Log all configuration parameters
    logger.info(f"🔧 Configuration (FIXED VERSION):")
    logger.info(f"   PPO Type: {args.type}")
    logger.info(f"   Data Path: {args.data}")
    logger.info(f"   Output Name: {args.output or 'default'}")
    logger.info(f"   Learning Rate: {args.learning_rate}")
    logger.info(f"   Episode Length: {args.episode_length}")
    logger.info(f"   Max Episodes: {args.episodes or 'unlimited'}")
    logger.info(f"   Max Minutes: {args.minutes or 'unlimited'}")
    logger.info("=" * 80)
    
    # Create trainer
    trainer = PPOTrainer(
        ppo_type=args.type,
        data_path=args.data,
        output_name=args.output,
        learning_rate=args.learning_rate,
        episode_length=args.episode_length
    )
    
    # Train model
    results = trainer.train(
        max_episodes=args.episodes,
        max_minutes=args.minutes
    )
    
    # Log final results
    logger.info("=" * 80)
    if results['success']:
        logger.info(f"🎉 Training completed successfully! (FIXED VERSION)")
        logger.info(f"📊 Episodes: {results['total_episodes']}")
        logger.info(f"⏱️  Time: {results['training_time_minutes']:.1f} minutes")
        logger.info(f"📈 Final reward: {results['final_avg_reward']:.6f}")
        logger.info(f"🔄 Final avg trades: {results['final_avg_trades']:.1f}")
        logger.info(f"🤖 PPO type: {results['ppo_type']}")
        logger.info(f"💾 Model saved: {results['model_path']}")
        logger.info(f"📝 Log saved: {log_path}")
        
        # Print to console as well
        print(f"\n🎉 Training completed successfully! (FIXED VERSION)")
        print(f"📊 Episodes: {results['total_episodes']}")
        print(f"⏱️  Time: {results['training_time_minutes']:.1f} minutes")
        print(f"📈 Final reward: {results['final_avg_reward']:.4f}")
        print(f"🔄 Final avg trades: {results['final_avg_trades']:.1f}")
        print(f"🤖 PPO type: {results['ppo_type']}")
        print(f"💾 Model saved: {results['model_path']}")
        print(f"📝 Log saved: {log_path}")
        return 0
    else:
        logger.error(f"❌ Training failed!")
        logger.error(f"Error: {results.get('reason', 'Unknown error')}")
        print(f"\n❌ Training failed: {results.get('reason', 'Unknown error')}")
        print(f"📝 Log saved: {log_path}")
        return 1

if __name__ == "__main__":
    exit(main())
