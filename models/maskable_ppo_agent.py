#!/usr/bin/env python3
"""
Maskable PPO Trading Agent
Advanced PPO agent with action masking for constrained trading environments.

Features:
- Dynamic action masking for risk management
- Enhanced policy network architecture
- Comprehensive training callbacks and monitoring
- Performance comparison utilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from datetime import datetime
import pickle
import json
from pathlib import Path

# Stable Baselines3 imports
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Maskable PPO imports
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Local imports
from models.maskable_trading_env import MaskableTradingEnv

logger = logging.getLogger(__name__)

class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for trading data with enhanced architecture.
    
    Processes market data, technical indicators, and portfolio state
    through specialized neural network layers.
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimensions
        n_input_features = observation_space.shape[0]
        
        # Market data processing layers
        self.market_net = nn.Sequential(
            nn.Linear(n_input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Technical indicators processing
        self.technical_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Portfolio state processing
        self.portfolio_net = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Final combination layer
        self.output_net = nn.Sequential(
            nn.Linear(16, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor"""
        
        # Process through market data layers
        market_features = self.market_net(observations)
        
        # Process through technical indicators
        tech_features = self.technical_net(market_features)
        
        # Process through portfolio state
        portfolio_features = self.portfolio_net(tech_features)
        
        # Final output
        output = self.output_net(portfolio_features)
        
        return output

class TradingMaskablePolicy(MaskableActorCriticPolicy):
    """
    Custom maskable policy for trading with enhanced architecture.
    """
    
    def __init__(self, *args, **kwargs):
        # Use custom feature extractor
        kwargs['features_extractor_class'] = TradingFeaturesExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 256}
        
        # Enhanced network architecture
        kwargs['net_arch'] = [
            {'pi': [256, 128, 64], 'vf': [256, 128, 64]}
        ]
        
        super().__init__(*args, **kwargs)

class TradingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress and performance.
    """
    
    def __init__(self, eval_freq: int = 1000, save_freq: int = 5000, 
                 save_path: str = "models/", verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        self.trade_counts = []
        
        # Best model tracking
        self.best_mean_reward = -np.inf
        self.best_model_path = None
        
    def _on_step(self) -> bool:
        """Called at each step during training"""
        
        # Log training progress
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_performance()
        
        # Save model periodically
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        
        return True
    
    def _evaluate_performance(self):
        """Evaluate current model performance"""
        
        # Get recent episode statistics
        if hasattr(self.training_env, 'get_attr'):
            try:
                # Get episode info from vectorized environment
                episode_infos = []
                for env in self.training_env.envs:
                    if hasattr(env, 'episode_info'):
                        episode_infos.append(env.episode_info)
                
                if episode_infos:
                    recent_rewards = [info.get('total_return', 0) for info in episode_infos[-10:]]
                    recent_trades = [info.get('total_trades', 0) for info in episode_infos[-10:]]
                    
                    if recent_rewards:
                        mean_reward = np.mean(recent_rewards)
                        mean_trades = np.mean(recent_trades)
                        
                        # Log performance
                        if self.verbose > 0:
                            print(f"Step {self.n_calls}: Mean Reward: {mean_reward:.4f}, "
                                  f"Mean Trades: {mean_trades:.1f}")
                        
                        # Update best model
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            self._save_best_model()
                        
                        # Store metrics
                        self.episode_rewards.append(mean_reward)
                        self.trade_counts.append(mean_trades)
                        
            except Exception as e:
                logger.warning(f"Error evaluating performance: {e}")
    
    def _save_model(self):
        """Save current model"""
        try:
            model_path = self.save_path / f"maskable_ppo_step_{self.n_calls}.zip"
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"Model saved to {model_path}")
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _save_best_model(self):
        """Save best performing model"""
        try:
            best_path = self.save_path / "best_maskable_ppo.zip"
            self.model.save(best_path)
            self.best_model_path = best_path
            
            if self.verbose > 0:
                print(f"New best model saved! Reward: {self.best_mean_reward:.4f}")
                
        except Exception as e:
            logger.error(f"Error saving best model: {e}")

class MaskablePPOAgent:
    """
    Maskable PPO Trading Agent with comprehensive training and evaluation capabilities.
    """
    
    def __init__(self, 
                 env_config: Dict[str, Any] = None,
                 model_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None):
        
        # Default configurations
        self.env_config = env_config or {
            'initial_capital': 100000,
            'max_position': 1000,
            'transaction_cost': 0.001,
            'max_drawdown_limit': 0.20,
            'lookback_window': 50
        }
        
        self.model_config = model_config or {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': {
                'net_arch': [{'pi': [256, 128, 64], 'vf': [256, 128, 64]}]
            }
        }
        
        self.training_config = training_config or {
            'total_timesteps': 100000,
            'eval_freq': 1000,
            'save_freq': 5000,
            'log_interval': 100
        }
        
        # Initialize components
        self.model = None
        self.env = None
        self.training_history = []
        
    def create_environment(self, data: pd.DataFrame, seed: int = 42) -> MaskableTradingEnv:
        """Create trading environment with action masking"""
        
        # Create base environment
        env = MaskableTradingEnv(data, **self.env_config)
        
        # Wrap with action masker
        def mask_fn(env):
            return env.get_action_mask()
        
        masked_env = ActionMasker(env, mask_fn)
        
        # Wrap with monitor for logging
        monitored_env = Monitor(masked_env)
        
        # Set random seed
        set_random_seed(seed)
        
        return monitored_env
    
    def create_model(self, env, seed: int = 42) -> MaskablePPO:
        """Create Maskable PPO model"""
        
        # Set random seed
        set_random_seed(seed)
        
        # Create model with custom policy
        model = MaskablePPO(
            TradingMaskablePolicy,
            env,
            verbose=1,
            seed=seed,
            **self.model_config
        )
        
        return model
    
    def train(self, 
              data: pd.DataFrame, 
              save_path: str = "models/",
              seed: int = 42) -> Dict[str, Any]:
        """
        Train the Maskable PPO agent.
        
        Args:
            data: Training data
            save_path: Path to save models
            seed: Random seed
            
        Returns:
            Training results and metrics
        """
        
        print("🎭 Starting Maskable PPO Training")
        print("=" * 50)
        
        start_time = datetime.now()
        
        # Create environment
        print("📊 Creating training environment...")
        self.env = self.create_environment(data, seed)
        
        # Vectorize environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Create model
        print("🧠 Creating Maskable PPO model...")
        self.model = self.create_model(vec_env, seed)
        
        # Create callback
        callback = TradingCallback(
            eval_freq=self.training_config['eval_freq'],
            save_freq=self.training_config['save_freq'],
            save_path=save_path,
            verbose=1
        )
        
        # Train model
        print(f"🚀 Training for {self.training_config['total_timesteps']:,} timesteps...")
        
        self.model.learn(
            total_timesteps=self.training_config['total_timesteps'],
            callback=callback,
            log_interval=self.training_config['log_interval'],
            progress_bar=True
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save final model
        final_model_path = Path(save_path) / "final_maskable_ppo.zip"
        self.model.save(final_model_path)
        
        # Compile training results
        results = {
            'training_time': training_time,
            'total_timesteps': self.training_config['total_timesteps'],
            'final_model_path': str(final_model_path),
            'best_model_path': str(callback.best_model_path) if callback.best_model_path else None,
            'best_mean_reward': callback.best_mean_reward,
            'episode_rewards': callback.episode_rewards,
            'trade_counts': callback.trade_counts,
            'env_config': self.env_config,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        # Save training results
        results_path = Path(save_path) / "maskable_ppo_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Training completed in {training_time:.1f} seconds")
        print(f"📈 Best mean reward: {callback.best_mean_reward:.4f}")
        print(f"💾 Models saved to: {save_path}")
        
        return results
    
    def evaluate(self, 
                 data: pd.DataFrame, 
                 model_path: str = None,
                 n_episodes: int = 10,
                 seed: int = 42) -> Dict[str, Any]:
        """
        Evaluate trained model performance.
        
        Args:
            data: Evaluation data
            model_path: Path to saved model (if None, uses current model)
            n_episodes: Number of evaluation episodes
            seed: Random seed
            
        Returns:
            Evaluation results and metrics
        """
        
        print("📊 Evaluating Maskable PPO Performance")
        print("-" * 40)
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            print(f"📂 Loading model from {model_path}")
            self.model = MaskablePPO.load(model_path)
        elif self.model is None:
            raise ValueError("No model available for evaluation")
        
        # Create evaluation environment
        eval_env = self.create_environment(data, seed)
        
        # Run evaluation episodes
        episode_results = []
        
        for episode in range(n_episodes):
            print(f"🎮 Running episode {episode + 1}/{n_episodes}")
            
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Get action mask
                action_mask = eval_env.get_action_mask()
                
                # Predict action
                action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
                
                # Take step
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            # Store episode results
            episode_info = info.copy()
            episode_info.update({
                'episode': episode + 1,
                'episode_reward': episode_reward,
                'episode_length': episode_length
            })
            episode_results.append(episode_info)
            
            print(f"   Episode {episode + 1} - Return: {episode_info.get('total_return', 0):.2%}, "
                  f"Trades: {episode_info.get('total_trades', 0)}")
        
        # Calculate aggregate metrics
        total_returns = [ep.get('total_return', 0) for ep in episode_results]
        sharpe_ratios = [ep.get('sharpe_ratio', 0) for ep in episode_results]
        max_drawdowns = [ep.get('max_drawdown', 0) for ep in episode_results]
        trade_counts = [ep.get('total_trades', 0) for ep in episode_results]
        win_rates = [ep.get('win_rate', 0) for ep in episode_results]
        
        evaluation_results = {
            'n_episodes': n_episodes,
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'mean_trades': np.mean(trade_counts),
            'mean_win_rate': np.mean(win_rates),
            'episode_results': episode_results,
            'evaluation_data_length': len(data)
        }
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Mean Return: {evaluation_results['mean_return']:.2%}")
        print(f"   Mean Sharpe: {evaluation_results['mean_sharpe']:.2f}")
        print(f"   Mean Drawdown: {evaluation_results['mean_max_drawdown']:.2%}")
        print(f"   Mean Trades: {evaluation_results['mean_trades']:.1f}")
        print(f"   Mean Win Rate: {evaluation_results['mean_win_rate']:.1%}")
        
        return evaluation_results
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model:
            self.model.save(path)
            print(f"💾 Model saved to {path}")
        else:
            print("❌ No model to save")
    
    def load_model(self, path: str):
        """Load trained model"""
        if Path(path).exists():
            self.model = MaskablePPO.load(path)
            print(f"📂 Model loaded from {path}")
        else:
            print(f"❌ Model file not found: {path}")

def create_comparison_framework():
    """
    Create framework for fair comparison between Standard PPO and Maskable PPO.
    """
    
    comparison_config = {
        'time_budget_minutes': 30,  # Fair time comparison
        'evaluation_episodes': 10,
        'random_seeds': [42, 123, 456],  # Multiple seeds for statistical significance
        'data_split': {
            'train_ratio': 0.7,
            'validation_ratio': 0.15,
            'test_ratio': 0.15
        }
    }
    
    return comparison_config

if __name__ == "__main__":
    # Test the Maskable PPO agent
    from data.data_manager import load_market_data
    
    print("🧪 Testing Maskable PPO Agent")
    
    # Load test data
    data = load_market_data()
    train_data = data.tail(2000)
    
    # Create agent
    agent = MaskablePPOAgent()
    
    # Quick training test (reduced timesteps for testing)
    agent.training_config['total_timesteps'] = 5000
    agent.training_config['eval_freq'] = 500
    agent.training_config['save_freq'] = 2000
    
    # Train agent
    results = agent.train(train_data, save_path="models/test/")
    
    # Evaluate agent
    eval_data = data.tail(500)
    eval_results = agent.evaluate(eval_data, n_episodes=3)
    
    print("🎉 Maskable PPO Agent test completed successfully!")
    print(f"📊 Training time: {results['training_time']:.1f}s")
    print(f"📈 Evaluation return: {eval_results['mean_return']:.2%}")
