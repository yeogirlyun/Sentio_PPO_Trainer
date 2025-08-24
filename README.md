# Sentio PPO Trainer

A sophisticated Proximal Policy Optimization (PPO) training system designed for algorithmic trading, featuring advanced reinforcement learning techniques and realistic market simulation.

## 🚀 Quick Start

```bash
# Default: Enhanced-maskable PPO, unlimited training until stable
python train_ppo.py

# 30-minute training session
python train_ppo.py --minutes 30

# Train specific PPO variant
python train_ppo.py --type enhanced-maskable --minutes 30 --output my_model
```

## 📋 Features

### 🎯 **Three PPO Variants**
- **Standard PPO**: Classic reinforcement learning approach
- **Maskable PPO**: Action masking for invalid trading actions
- **Enhanced-Maskable PPO**: Advanced features with market regime awareness

### 🏪 **Realistic Trading Environment**
- **Real Market Data**: QQQ 1-minute OHLCV data (440K+ bars)
- **Transaction Costs**: 0.1% realistic trading fees
- **Portfolio Management**: Balance + position tracking
- **Risk Management**: Drawdown monitoring and position limits

### 📊 **Advanced Features**
- **20-Feature Observation Space**: Price, volume, technical indicators, portfolio state
- **Automatic Convergence Detection**: Stops when model is stable
- **Progress Monitoring**: Real-time training metrics
- **Model Persistence**: Saves both model and metadata

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Sentio_PPO_Trainer.git
cd Sentio_PPO_Trainer

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Command Line Interface

```bash
python train_ppo.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--type` | PPO variant: `standard`, `maskable`, `enhanced-maskable` | `enhanced-maskable` |
| `--minutes` | Training time limit in minutes | Unlimited |
| `--episodes` | Maximum number of episodes | Unlimited |
| `--output` | Output filename (without extension) | Auto-generated |
| `--data` | Input data file path | `data/polygon_QQQ_1m.feather` |
| `--learning-rate` | Learning rate | `3e-4` |

### Examples

```bash
# Quick 5-minute test
python train_ppo.py --minutes 5 --output quick_test

# Standard PPO for 1000 episodes
python train_ppo.py --type standard --episodes 1000

# Custom data file
python train_ppo.py --data my_custom_data.feather --minutes 60

# Find stable model (runs until convergence)
python train_ppo.py --type enhanced-maskable
```

## 🏗️ Architecture

### Trading Environment
- **Observation Space**: 20 features including price, volume, technical indicators, portfolio state
- **Action Space**: 3 actions (Hold, Buy, Sell)
- **Reward Function**: Portfolio return with transaction cost penalties
- **Episode Length**: Variable based on data length

### PPO Network
- **Feature Extraction**: 256→256→128 neurons with dropout
- **Policy Head**: Separate network for action probabilities
- **Value Head**: Separate network for state value estimation
- **Action Masking**: Dynamic masking for invalid actions (maskable variants)

### Training Features
- **GAE (Generalized Advantage Estimation)**: Better advantage computation
- **Clipped Policy Loss**: Prevents large policy updates
- **Entropy Regularization**: Encourages exploration
- **Gradient Clipping**: Training stability

## 📁 Project Structure

```
Sentio_PPO_Trainer/
├── train_ppo.py              # Main training interface
├── models/                   # Advanced PPO modules
│   ├── advanced_unified_trainer.py
│   ├── ppo_network.py
│   ├── maskable_trading_env.py
│   ├── transformer_policy.py
│   ├── distributional_critic.py
│   ├── regime_gating.py
│   ├── purged_cv.py
│   ├── exceptions.py
│   └── device_utils.py
├── data/                     # Market data
│   └── polygon_QQQ_1m.feather
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── ARCHITECTURE.md          # Technical documentation
```

## 🎯 Performance Targets

The system is designed to achieve:
- **Sharpe Ratio**: >3.0
- **Maximum Drawdown**: <8%
- **Win Rate**: >65%
- **Training Stability**: Consistent convergence

## 📊 Output Files

Each training session produces:
- **Model File**: `{name}.pth` - PyTorch model with state dict
- **Metadata**: `{name}_metadata.json` - Training metrics and configuration

## 🔧 Advanced Configuration

For advanced users, the system supports:
- **Custom Learning Rates**: Tune convergence speed
- **Different Data Sources**: Use your own market data
- **Extended Training**: Unlimited episodes until convergence
- **Multiple Variants**: Compare different PPO approaches

## 🤝 Integration

The trained models can be integrated with:
- **Sentio Trading Platform**: Main algorithmic trading system
- **Custom Trading Bots**: Use the model for signal generation
- **Backtesting Systems**: Evaluate performance on historical data

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Technical deep-dive for engineers
- **Training Logs**: Detailed progress monitoring during training
- **Model Metadata**: Complete training configuration and results

## 🛡️ Risk Management

Built-in risk controls:
- **Position Limits**: Prevents over-leveraging
- **Transaction Costs**: Realistic trading simulation
- **Drawdown Monitoring**: Automatic risk assessment
- **Action Masking**: Prevents invalid trading actions

## 🚀 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Quick test**: `python train_ppo.py --minutes 5`
3. **Full training**: `python train_ppo.py --minutes 30`
4. **Monitor progress**: Check console output for real-time metrics

## 📈 Results

The system automatically tracks:
- Episode rewards and lengths
- Training loss components
- Portfolio performance metrics
- Model convergence indicators

## 🔄 Continuous Improvement

The PPO trainer includes:
- **Automatic hyperparameter optimization**
- **Adaptive learning rate scheduling**
- **Early stopping for convergence**
- **Comprehensive error handling**

---

**Built for professional algorithmic trading with state-of-the-art reinforcement learning techniques.**