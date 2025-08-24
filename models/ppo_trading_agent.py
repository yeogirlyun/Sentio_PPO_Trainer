#!/usr/bin/env python3
"""
Sentio Trader PPO (Proximal Policy Optimization) Neural Network Agent
Deep reinforcement learning agent for trading decisions
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. PPO agent will use placeholder implementation.")

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("Gymnasium/Gym not available. Using placeholder environment.")

from data.data_manager import SentioDataManager

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO agent"""
    # Network architecture
    hidden_size: int = 256
    num_layers: int = 3
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 10
    max_grad_norm: float = 0.5
    
    # Environment parameters
    lookback_window: int = 240  # 4 hours of minute data
    action_space_size: int = 3  # Buy, Hold, Sell
    
    # Feature engineering
    use_technical_indicators: bool = True
    normalize_features: bool = True


class TradingEnvironment:
    """Trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, config: PPOConfig):
        """
        Initialize trading environment
        
        Args:
            data: Market data (OHLCV)
            config: PPO configuration
        """
        self.data = data
        self.config = config
        self.current_step = 0
        self.max_steps = len(data) - config.lookback_window
        
        # Trading state
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.cash = 100000  # Starting cash
        self.shares = 0
        self.portfolio_value = self.cash
        
        # Action and observation spaces
        if GYM_AVAILABLE:
            self.action_space = spaces.Discrete(config.action_space_size)
            
            # Observation space: OHLCV + technical indicators + portfolio state
            obs_size = self._calculate_observation_size()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(obs_size,), dtype=np.float32
            )
        
        # Feature engineering
        self._prepare_features()
        
        logger.info(f"TradingEnvironment initialized with {len(data)} data points")
    
    def _calculate_observation_size(self) -> int:
        """Calculate size of observation space"""
        # Prepare features first to get accurate count
        if not hasattr(self, 'features'):
            self._prepare_features()
        
        # Count actual feature columns (market data)
        feature_count = len(self.features.columns)
        
        # Portfolio state features (repeated for each timestep)
        portfolio_features = 4  # position, cash_ratio, portfolio_value_change, unrealized_pnl
        
        # Total: (market features + portfolio features) × lookback_window
        return (feature_count + portfolio_features) * self.config.lookback_window
    
    def _prepare_features(self):
        """Prepare and engineer features"""
        self.features = self.data.copy()
        
        if self.config.use_technical_indicators:
            self._add_technical_indicators()
        
        if self.config.normalize_features:
            self._normalize_features()
    
    def _add_technical_indicators(self):
        """Add technical indicators to features"""
        # RSI
        delta = self.features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.features['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        self.features['sma_20'] = self.features['close'].rolling(20).mean()
        self.features['sma_50'] = self.features['close'].rolling(50).mean()
        self.features['ema_12'] = self.features['close'].ewm(span=12).mean()
        self.features['ema_26'] = self.features['close'].ewm(span=26).mean()
        
        # MACD
        self.features['macd'] = self.features['ema_12'] - self.features['ema_26']
        self.features['macd_signal'] = self.features['macd'].ewm(span=9).mean()
        self.features['macd_histogram'] = self.features['macd'] - self.features['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = self.features['close'].rolling(bb_period).mean()
        std = self.features['close'].rolling(bb_period).std()
        self.features['bb_upper'] = sma + (std * bb_std)
        self.features['bb_lower'] = sma - (std * bb_std)
        self.features['bb_width'] = (self.features['bb_upper'] - self.features['bb_lower']) / sma
        
        # Volume indicators
        self.features['volume_sma'] = self.features['volume'].rolling(20).mean()
        self.features['volume_ratio'] = self.features['volume'] / self.features['volume_sma']
        
        # Price momentum
        self.features['price_change_1'] = self.features['close'].pct_change(1)
        self.features['price_change_5'] = self.features['close'].pct_change(5)
        self.features['price_change_20'] = self.features['close'].pct_change(20)
        
        # Volatility
        self.features['volatility'] = self.features['close'].pct_change().rolling(20).std()
        
        # Fill NaN values (pandas 2.0+ compatibility)
        self.features.ffill(inplace=True)
        self.features.fillna(0, inplace=True)
    
    def _normalize_features(self):
        """Normalize features to [-1, 1] range"""
        # Skip normalization for now - would need proper scaling
        pass
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        self.current_step = 0
        self.position = 0
        self.cash = self.config.initial_capital
        self.shares = 0
        self.portfolio_value = self.config.initial_capital
        
        # Track episode start for return calculation
        self.episode_start_value = self.config.initial_capital
        
        return self._get_observation()
    
    def get_episode_return(self) -> float:
        """Get the total return for the current episode"""
        if self.episode_start_value == 0:
            return 0.0
        return (self.portfolio_value - self.episode_start_value) / self.episode_start_value

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take (0: sell, 1: hold, 2: buy)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
        
        # Get current price
        current_price = self.features.iloc[self.current_step + self.config.lookback_window]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'shares': self.shares
        }
        
        return obs, reward, done, info
    
    def _execute_action(self, action: int, price: float) -> float:
        """
        Execute trading action and calculate reward
        
        Args:
            action: Trading action
            price: Current price
            
        Returns:
            Reward for the action
        """
        prev_portfolio_value = self.portfolio_value
        
        # Action mapping: 0=sell, 1=hold, 2=buy
        if action == 0 and self.position > 0:  # Sell
            self.cash += self.shares * price
            self.shares = 0
            self.position = 0
            
        elif action == 2 and self.position <= 0:  # Buy
            shares_to_buy = int(self.cash * 0.95 / price)  # Use 95% of cash
            if shares_to_buy > 0:
                self.cash -= shares_to_buy * price
                self.shares = shares_to_buy
                self.position = 1
        
        # Calculate new portfolio value
        self.portfolio_value = self.cash + (self.shares * price)
        
        # Calculate reward as normalized portfolio value change
        # Use basis points (0.01%) to keep rewards in reasonable range
        reward = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value * 10000  # Convert to basis points
        reward = max(-100, min(100, reward))  # Clip to reasonable range [-100, 100] basis points
        
        # Add penalty for excessive trading
        if action != 1:  # Not holding
            reward -= 0.001  # Small transaction cost
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.current_step + self.config.lookback_window >= len(self.features):
            # Return zeros if we're at the end
            return np.zeros(self._calculate_observation_size(), dtype=np.float32)
        
        # Get historical data window
        start_idx = self.current_step
        end_idx = self.current_step + self.config.lookback_window
        
        window_data = self.features.iloc[start_idx:end_idx]
        
        # Extract OHLCV features
        ohlcv_features = window_data[['open', 'high', 'low', 'close', 'volume']].values.flatten()
        
        # Add technical indicators if enabled
        if self.config.use_technical_indicators:
            tech_columns = [col for col in window_data.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume']]
            tech_features = window_data[tech_columns].values.flatten()
            ohlcv_features = np.concatenate([ohlcv_features, tech_features])
        
        # Add portfolio state features
        current_price = window_data['close'].iloc[-1]
        portfolio_features = np.array([
            self.position,
            self.cash / 100000,  # Normalized cash
            (self.portfolio_value - 100000) / 100000,  # Normalized portfolio change
            (self.shares * current_price - self.shares * window_data['close'].iloc[0]) / 100000 if self.shares > 0 else 0  # Unrealized PnL
        ])
        
        # Repeat portfolio features for each time step in window
        portfolio_features_repeated = np.tile(portfolio_features, self.config.lookback_window)
        
        # Combine all features
        observation = np.concatenate([ohlcv_features, portfolio_features_repeated]).astype(np.float32)
        
        return observation


class PPONetwork(nn.Module):
    """PPO Actor-Critic Network"""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int, action_size: int):
        """
        Initialize PPO network
        
        Args:
            input_size: Size of input observations
            hidden_size: Size of hidden layers
            num_layers: Number of hidden layers
            action_size: Size of action space
        """
        super(PPONetwork, self).__init__()
        
        # Shared layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """Forward pass"""
        shared = self.shared_layers(x)
        
        # Policy logits
        policy_logits = self.actor(shared)
        
        # State value
        value = self.critic(shared)
        
        return policy_logits, value
    
    def get_action_and_value(self, x, action=None):
        """Get action and value for PPO training"""
        policy_logits, value = self.forward(x)
        
        # Create action distribution
        probs = Categorical(logits=policy_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


class PPOAgent:
    """PPO Trading Agent"""
    
    def __init__(self, config: PPOConfig, observation_size: int):
        """
        Initialize PPO agent
        
        Args:
            config: PPO configuration
            observation_size: Size of observation space
        """
        self.config = config
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using placeholder PPO agent.")
            return
        
        # Initialize network
        self.network = PPONetwork(
            input_size=observation_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            action_size=config.action_space_size
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # Training data storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        logger.info("PPO Agent initialized")
    
    def get_action(self, observation: np.ndarray) -> Tuple[int, float, float]:
        """
        Get action from current policy
        
        Args:
            observation: Current observation
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        if not TORCH_AVAILABLE:
            # Random action for placeholder
            return np.random.randint(0, self.config.action_space_size), 0.0, 0.0
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action, log_prob, entropy, value = self.network.get_action_and_value(obs_tensor)
            
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, obs: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool):
        """Store transition for training"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def train(self) -> Dict[str, float]:
        """Train the PPO agent"""
        if not TORCH_AVAILABLE or len(self.observations) == 0:
            return {}
        
        # Convert to tensors
        observations = torch.FloatTensor(np.array(self.observations))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        values = torch.FloatTensor(self.values)
        
        # Calculate advantages using GAE
        advantages, returns = self._calculate_gae(self.rewards, values, self.dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_loss_sum = 0
        
        for epoch in range(self.config.num_epochs):
            # Get current policy outputs
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                observations, actions
            )
            
            # Calculate policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                               1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Calculate entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.config.value_coef * value_loss + 
                   self.config.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 
                                         self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_loss_sum += entropy_loss.item()
        
        # Clear stored transitions
        self._clear_memory()
        
        return {
            'total_loss': total_loss / self.config.num_epochs,
            'policy_loss': policy_loss_sum / self.config.num_epochs,
            'value_loss': value_loss_sum / self.config.num_epochs,
            'entropy_loss': entropy_loss_sum / self.config.num_epochs
        }
    
    def _calculate_gae(self, rewards: List[float], values: List[float], 
                      dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        
        return advantages, returns
    
    def _clear_memory(self):
        """Clear stored transitions"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if TORCH_AVAILABLE:
            torch.save(self.network.state_dict(), filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        if TORCH_AVAILABLE:
            self.network.load_state_dict(torch.load(filepath))
            logger.info(f"Model loaded from {filepath}")


class SentioPPOTrainer:
    """High-level trainer for PPO trading agent"""
    
    def __init__(self, config: Optional[PPOConfig] = None):
        """
        Initialize PPO trainer
        
        Args:
            config: PPO configuration (uses default if None)
        """
        self.config = config or PPOConfig()
        self.data_manager = SentioDataManager()
        
        logger.info("SentioPPOTrainer initialized")
    
    def train_agent(self, num_episodes: int = 1000, 
                   save_interval: int = 100) -> Dict[str, Any]:
        """
        Train PPO agent on market data
        
        Args:
            num_episodes: Number of training episodes
            save_interval: Save model every N episodes
            
        Returns:
            Training results
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available. Cannot train PPO agent.")
            return {'error': 'PyTorch not available'}
        
        # Load market data
        data = self.data_manager.load_market_data()
        
        # Create environment
        env = TradingEnvironment(data, self.config)
        
        # Create agent
        agent = PPOAgent(self.config, env._calculate_observation_size())
        
        # Training setup
        episode_rewards = []
        training_losses = []
        start_time = datetime.now()
        
        # Progress tracking
        print(f"\n🚀 Starting PPO Training")
        print(f"📊 Episodes: {num_episodes}")
        print(f"🎯 Observation size: {env._calculate_observation_size()}")
        print(f"🔄 Lookback window: {self.config.lookback_window}")
        print(f"💾 Save interval: {save_interval} episodes")
        print(f"⏰ Started at: {start_time.strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Quick single episode benchmark for immediate estimate
        print("🔍 Running quick benchmark episode...")
        quick_start = datetime.now()
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:  # Limit to 100 steps for quick estimate
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        quick_time = (datetime.now() - quick_start).total_seconds()
        
        # Rough estimate (assuming full episodes are ~10x longer than 100 steps)
        estimated_episode_time = quick_time * (env.max_steps / max(steps, 1))
        rough_total_time = estimated_episode_time * num_episodes
        rough_completion = datetime.now() + pd.Timedelta(seconds=rough_total_time)
        
        print(f"⚡ **QUICK ESTIMATE** (based on {steps} steps)")
        print(f"📊 Rough episode time: ~{estimated_episode_time:.1f} seconds")
        print(f"🕐 Rough total time: ~{rough_total_time/3600:.1f} hours ({rough_total_time/60:.0f} minutes)")
        print(f"🎯 Rough completion: ~{rough_completion.strftime('%H:%M:%S')}")
        print(f"📝 Note: This is a rough estimate, will be refined after 3 full episodes")
        print("-" * 60)
        
        # Reset environment for actual training
        env.reset()
        
        # More accurate benchmark tracking
        benchmark_episodes = min(3, num_episodes)
        benchmark_start = datetime.now()
        
        # Training loop
        for episode in range(num_episodes):
            episode_start = datetime.now()
            obs = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done:
                # Get action
                action, log_prob, value = agent.get_action(obs)
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                
                # Store transition
                agent.store_transition(obs, action, reward, value, log_prob, done)
                
                obs = next_obs
                episode_reward += reward
                steps += 1
            
            # Train agent
            if len(agent.observations) >= self.config.batch_size:
                losses = agent.train()
                training_losses.append(losses)
            
            episode_rewards.append(episode_reward)
            
            # Early time estimation after benchmark episodes
            if episode + 1 == benchmark_episodes:
                benchmark_time = datetime.now() - benchmark_start
                avg_benchmark_time = benchmark_time.total_seconds() / benchmark_episodes
                estimated_total_seconds = avg_benchmark_time * num_episodes
                estimated_total_time = pd.Timedelta(seconds=estimated_total_seconds)
                estimated_completion = datetime.now() + estimated_total_time
                
                print(f"\n⏱️  **TIME ESTIMATE** (based on {benchmark_episodes} episodes)")
                print(f"📊 Average time per episode: {avg_benchmark_time:.1f} seconds")
                print(f"🕐 Estimated total training time: {str(estimated_total_time).split('.')[0]}")
                print(f"🎯 Estimated completion: {estimated_completion.strftime('%H:%M:%S')}")
                print(f"📈 Benchmark episodes avg reward: {np.mean(episode_rewards[-benchmark_episodes:]):.2f}")
                print("=" * 60)
            
            # Progress reporting (more frequent for early episodes)
            show_progress = (
                episode < 10 or  # Show first 10 episodes
                episode % 10 == 0 or  # Every 10th episode
                episode == num_episodes - 1  # Final episode
            )
            
            if show_progress:
                # Calculate progress metrics
                progress_pct = (episode + 1) / num_episodes * 100
                elapsed = datetime.now() - start_time
                
                # Time estimates
                if episode > 0:
                    avg_episode_time = elapsed.total_seconds() / (episode + 1)
                    remaining_episodes = num_episodes - (episode + 1)
                    eta_seconds = remaining_episodes * avg_episode_time
                    eta = datetime.now() + pd.Timedelta(seconds=eta_seconds)
                else:
                    eta = "Calculating..."
                
                # Performance metrics
                avg_reward = np.mean(episode_rewards[-10:])
                best_reward = max(episode_rewards) if episode_rewards else 0
                
                # Loss metrics
                if training_losses:
                    latest_loss = training_losses[-1]
                    avg_policy_loss = latest_loss.get('policy_loss', 0)
                    avg_value_loss = latest_loss.get('value_loss', 0)
                    loss_str = f"Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}"
                else:
                    loss_str = "No training yet"
                
                print(f"📈 Episode {episode:4d}/{num_episodes} ({progress_pct:5.1f}%) | "
                      f"Reward: {episode_reward:8.2f} (avg: {avg_reward:6.2f}, best: {best_reward:6.2f}) | "
                      f"Steps: {steps:3d} | Loss: {loss_str}")
                
                if episode > 0 and isinstance(eta, datetime):
                    print(f"⏱️  Elapsed: {str(elapsed).split('.')[0]} | ETA: {eta.strftime('%H:%M:%S')} | "
                          f"Avg: {avg_episode_time:.1f}s/episode")
                
                if episode % 50 == 0 and episode > 0:
                    print("-" * 60)
            
            # Save model
            if episode % save_interval == 0 and episode > 0:
                model_path = f"models/ppo_agent_episode_{episode}.pth"
                agent.save_model(model_path)
                print(f"💾 Model saved: {model_path}")
        
        # Final summary
        total_time = datetime.now() - start_time
        final_avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        print("\n" + "=" * 60)
        print(f"🎯 Training Complete!")
        print(f"⏰ Total time: {str(total_time).split('.')[0]}")
        print(f"📊 Final average reward: {final_avg_reward:.4f}")
        print(f"🏆 Best episode reward: {max(episode_rewards):.4f}")
        print(f"📈 Total episodes: {len(episode_rewards)}")
        print(f"🧠 Training iterations: {len(training_losses)}")
        
        # Save final model
        final_model_path = "models/ppo_agent_final.pth"
        agent.save_model(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        return {
            'episode_rewards': episode_rewards,
            'training_losses': training_losses,
            'final_avg_reward': final_avg_reward,
            'best_reward': max(episode_rewards),
            'total_time_seconds': total_time.total_seconds(),
            'model_path': final_model_path
        }


def main():
    """Main function for PPO training"""
    logging.basicConfig(level=logging.INFO)
    
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available. Install PyTorch to use PPO agent:")
        print("   pip install torch torchvision torchaudio")
        return
    
    # Create trainer
    config = PPOConfig()
    trainer = SentioPPOTrainer(config)
    
    # Train agent
    print("🤖 Starting PPO agent training...")
    results = trainer.train_agent(num_episodes=200, save_interval=50)
    
    print(f"\n🎯 Training completed!")
    print(f"📊 Final average reward: {results['final_avg_reward']:.4f}")


if __name__ == "__main__":
    main()
