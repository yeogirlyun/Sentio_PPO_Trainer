#!/usr/bin/env python3
"""
Unified PPO Training Interface
Simple, clean interface for training PPO models with different variants.

Usage:
    python train_ppo.py                                    # Enhanced-maskable PPO, unlimited training
    python train_ppo.py --minutes 30                      # 30-minute training
    python train_ppo.py --episodes 1000                   # 1000 episodes
    python train_ppo.py --type standard                   # Standard PPO
    python train_ppo.py --type maskable                   # Maskable PPO
    python train_ppo.py --output my_model                 # Custom output name
    python train_ppo.py --data custom_data.feather        # Custom data file
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEnvironment:
    """Realistic trading environment for PPO training."""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_trades = 0
        self.episode_reward = 0.0
        self.trade_history = []
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current market observation with 20 features."""
        if self.current_step >= len(self.data) - 1:
            return np.zeros(20, dtype=np.float32)
        
        current_idx = self.current_step
        lookback = min(20, current_idx)
        
        # Price features
        current_price = self.data.iloc[current_idx]['close']
        if lookback > 0:
            recent_prices = self.data.iloc[current_idx-lookback:current_idx]['close'].values
            price_change = (current_price - recent_prices[-1]) / recent_prices[-1] if len(recent_prices) > 0 else 0.0
            volatility = np.std(recent_prices) / np.mean(recent_prices) if len(recent_prices) > 1 else 0.0
        else:
            price_change = 0.0
            volatility = 0.0
        
        # Volume features
        current_volume = self.data.iloc[current_idx]['volume']
        if lookback > 0:
            recent_volumes = self.data.iloc[current_idx-lookback:current_idx]['volume'].values
            volume_ratio = current_volume / np.mean(recent_volumes) if len(recent_volumes) > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        # Technical indicators
        if lookback >= 10:
            prices = self.data.iloc[current_idx-lookback:current_idx]['close'].values
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
        drawdown = max(0, (self.initial_balance - portfolio_value) / self.initial_balance)
        
        # Create 20-feature observation
        obs = np.array([
            # Price features (6)
            current_price / 100.0,                           # Normalized price
            price_change,                                    # Price change
            volatility,                                      # Volatility
            (sma_short / current_price) - 1.0,              # Short SMA ratio
            (sma_long / current_price) - 1.0,               # Long SMA ratio
            (rsi / 100.0) - 0.5,                           # Normalized RSI
            
            # Volume features (2)
            np.log(max(1, current_volume)) / 15.0,          # Log volume
            np.clip(volume_ratio - 1.0, -2.0, 2.0),        # Volume ratio
            
            # Portfolio features (4)
            (self.balance / self.initial_balance) - 1.0,     # Balance change
            np.clip(position_ratio, -1.0, 1.0),            # Position ratio
            (portfolio_value / self.initial_balance) - 1.0,  # Total return
            drawdown,                                        # Drawdown
            
            # Market timing (4)
            self.current_step / len(self.data),              # Episode progress
            (self.current_step % 390) / 390.0,              # Intraday position
            np.sin(2 * np.pi * self.current_step / 390),    # Cyclical time
            np.cos(2 * np.pi * self.current_step / 390),    # Cyclical time
            
            # Trading features (4)
            min(1.0, self.total_trades / 100.0),            # Trade frequency
            1.0 if self.position > 0 else 0.0,              # Long position flag
            1.0 if self.position < 0 else 0.0,              # Short position flag
            1.0 if len(self.trade_history) > 0 and self.trade_history[-1]['profit'] > 0 else 0.0  # Last trade profit
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
        """Execute trading action: 0=hold, 1=buy, 2=sell."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, False, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0.0
        transaction_cost = 0.001  # 0.1% transaction cost
        
        # Execute action
        if action == 1 and self.balance > current_price * (1 + transaction_cost):
            # Buy action
            shares_to_buy = self.balance / (current_price * (1 + transaction_cost))
            cost = shares_to_buy * current_price * (1 + transaction_cost)
            self.balance -= cost
            old_position = self.position
            self.position += shares_to_buy
            self.total_trades += 1
            
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
            profit = proceeds - (self.position * self.trade_history[-1]['price'] if self.trade_history else 0)
            self.balance += proceeds
            self.position = 0
            self.total_trades += 1
            
            if self.trade_history:
                self.trade_history[-1]['profit'] = profit
            
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        if self.current_step < len(self.data):
            next_price = self.data.iloc[self.current_step]['close']
            portfolio_value = self.balance + self.position * next_price
            
            # Portfolio return reward
            portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance
            reward = portfolio_return * 100
            
            # Penalty for excessive trading
            if self.total_trades > self.current_step / 20:
                reward -= 0.01
            
            # Bonus for profitable trades
            if len(self.trade_history) > 0 and self.trade_history[-1]['profit'] > 0:
                reward += 0.1
        
        done = self.current_step >= len(self.data) - 1
        self.episode_reward += reward
        
        return self._get_observation(), reward, done, False, {
            'portfolio_value': self.balance + self.position * current_price,
            'position': self.position,
            'balance': self.balance,
            'total_trades': self.total_trades
        }

class PPONetwork(nn.Module):
    """PPO Network with support for different variants."""
    
    def __init__(self, input_size=20, hidden_size=256, action_size=3, ppo_type='enhanced-maskable'):
        super().__init__()
        self.ppo_type = ppo_type
        self.action_size = action_size
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        
        # Policy head
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
        
    def forward(self, x, action_mask=None):
        features = self.feature_net(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        
        # Apply action masking for maskable variants
        if ('maskable' in self.ppo_type or 'enhanced' in self.ppo_type) and action_mask is not None:
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
    """Unified PPO Trainer supporting all variants."""
    
    def __init__(self, 
                 ppo_type: str = 'enhanced-maskable',
                 data_path: str = 'data/polygon_QQQ_1m.feather',
                 output_name: Optional[str] = None,
                 learning_rate: float = 3e-4):
        
        self.ppo_type = ppo_type
        self.device = torch.device('cpu')  # CPU for compatibility
        
        # Load data
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} rows of market data")
        
        # Create environment
        self.env = TradingEnvironment(self.data)
        
        # Create network
        self.network = PPONetwork(ppo_type=ppo_type)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_start_time = None
        
        # Output naming
        if output_name:
            self.output_name = output_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_name = f"{ppo_type}_model_{timestamp}"
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load market data from file."""
        path = Path(data_path)
        if path.exists():
            df = pd.read_feather(path)
            # Use recent data for training (last 20k rows)
            return df.tail(20000).reset_index(drop=True)
        else:
            logger.warning(f"Data file {data_path} not found, generating synthetic data")
            # Generate realistic synthetic data
            n_points = 10000
            dates = pd.date_range('2023-01-01', periods=n_points, freq='1min')
            
            # Generate price with trend and volatility
            returns = np.random.normal(0.0001, 0.02, n_points)  # Small positive drift
            price = 100 * np.exp(np.cumsum(returns))
            
            return pd.DataFrame({
                'open': price + np.random.normal(0, 0.1, n_points),
                'high': price + np.abs(np.random.normal(0, 0.2, n_points)),
                'low': price - np.abs(np.random.normal(0, 0.2, n_points)),
                'close': price,
                'volume': np.random.lognormal(8, 1, n_points).astype(int)
            }, index=dates)
    
    def collect_rollout(self, num_steps: int = 2000):
        """Collect experience rollout."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        obs = self.env.reset()
        
        for step in range(num_steps):
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action mask for maskable variants
            action_mask = None
            if 'maskable' in self.ppo_type or 'enhanced' in self.ppo_type:
                current_price = self.env.data.iloc[self.env.current_step]['close']
                action_mask = self.network.get_action_mask(
                    obs, self.env.position, self.env.balance, current_price
                ).unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                logits, value = self.network(state_tensor, action_mask)
                probs = torch.softmax(logits, dim=-1)
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
        """Perform PPO update."""
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
              target_reward: float = 0.5) -> Dict:
        """Train the PPO model."""
        
        logger.info(f"Starting {self.ppo_type} PPO training")
        logger.info(f"Limits: {max_episodes or 'unlimited'} episodes, {max_minutes or 'unlimited'} minutes")
        logger.info(f"Target reward: {target_reward}")
        
        self.training_start_time = time.time()
        iteration = 0
        
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
                
                # Check if model is stable (for unlimited training)
                if not max_episodes and not max_minutes and len(self.episode_rewards) >= 100:
                    recent_rewards = self.episode_rewards[-50:]
                    avg_reward = np.mean(recent_rewards)
                    reward_std = np.std(recent_rewards)
                    
                    if avg_reward >= target_reward and reward_std < 0.1:
                        logger.info(f"Model converged! Avg reward: {avg_reward:.4f}, Std: {reward_std:.4f}")
                        break
                
                # Collect experience
                states, actions, rewards, values, log_probs, dones = self.collect_rollout(2000)
                
                # Compute advantages
                advantages, returns = self.compute_gae(rewards, values, dones)
                
                # PPO update
                loss_info = self.ppo_update(states, actions, log_probs, advantages, returns)
                
                # Log progress
                if len(self.episode_rewards) > 0 and iteration % 5 == 0:
                    recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                    avg_reward = np.mean(recent_rewards)
                    avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else 0
                    elapsed_time = (time.time() - self.training_start_time) / 60
                    
                    logger.info(f"Iteration {iteration} | Episodes: {len(self.episode_rewards)} | "
                              f"Avg Reward: {avg_reward:.4f} | Avg Length: {avg_length:.1f} | "
                              f"Time: {elapsed_time:.1f}min | Loss: {loss_info['total_loss']:.4f}")
            
            # Save model
            self._save_model()
            
            # Return results
            training_time = time.time() - self.training_start_time
            final_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0
            
            return {
                'success': True,
                'total_episodes': len(self.episode_rewards),
                'training_time_minutes': training_time / 60,
                'final_avg_reward': final_reward,
                'model_path': f"{self.output_name}.pth",
                'ppo_type': self.ppo_type
            }
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_model()
            return {'success': False, 'reason': 'interrupted'}
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _save_model(self):
        """Save the trained model."""
        model_path = f"{self.output_name}.pth"
        metadata_path = f"{self.output_name}_metadata.json"
        
        # Save model
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ppo_type': self.ppo_type,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }, model_path)
        
        # Save metadata
        metadata = {
            'ppo_type': self.ppo_type,
            'total_episodes': len(self.episode_rewards),
            'training_time_minutes': (time.time() - self.training_start_time) / 60 if self.training_start_time else 0,
            'final_avg_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
            'model_path': model_path,
            'created_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Unified PPO Training Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ppo.py                                    # Enhanced-maskable PPO, unlimited training
  python train_ppo.py --minutes 30                      # 30-minute training
  python train_ppo.py --episodes 1000                   # 1000 episodes
  python train_ppo.py --type standard                   # Standard PPO
  python train_ppo.py --type maskable                   # Maskable PPO
  python train_ppo.py --output my_model                 # Custom output name
  python train_ppo.py --data custom_data.feather        # Custom data file
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
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PPOTrainer(
        ppo_type=args.type,
        data_path=args.data,
        output_name=args.output,
        learning_rate=args.learning_rate
    )
    
    # Train model
    results = trainer.train(
        max_episodes=args.episodes,
        max_minutes=args.minutes
    )
    
    # Print results
    if results['success']:
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Episodes: {results['total_episodes']}")
        print(f"⏱️  Time: {results['training_time_minutes']:.1f} minutes")
        print(f"📈 Final reward: {results['final_avg_reward']:.4f}")
        print(f"🤖 PPO type: {results['ppo_type']}")
        print(f"💾 Model saved: {results['model_path']}")
        return 0
    else:
        print(f"\n❌ Training failed: {results.get('reason', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    exit(main())
