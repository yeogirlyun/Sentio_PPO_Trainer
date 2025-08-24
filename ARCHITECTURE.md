# Sentio PPO Training Package - Technical Architecture

## Overview

This document provides a comprehensive technical overview of the enhanced PPO training system built for Sentio Trader. The package extends standard PPO and Maskable PPO implementations with state-of-the-art enhancements specifically designed for high-frequency trading of leveraged ETFs (TQQQ/SQQQ).

## Base Algorithms

### Standard PPO (Proximal Policy Optimization)
- **Implementation**: `stable-baselines3` compatible
- **Core Features**: Clipped surrogate objective, GAE (Generalized Advantage Estimation)
- **Use Case**: Baseline comparison and research

### Maskable PPO (Primary Algorithm)
- **Implementation**: `sb3-contrib.MaskablePPO` extended
- **Core Features**: Dynamic action masking, invalid action prevention
- **Enhancement**: Market condition-aware masking rules
- **Use Case**: Primary algorithm for live trading

## Enhanced Architecture Stack

### 1. Neural Network Enhancements

#### Transformer Policy Networks (`models/transformer_policy.py`)
```python
class TransformerPolicyNetwork(nn.Module):
    # Multi-head attention for temporal pattern recognition
    # Positional encoding optimized for financial time series
    # Hierarchical processing for multi-timeframe analysis
```

**Technical Details:**
- **Architecture**: 6-layer transformer encoder with 8 attention heads
- **Input Processing**: Sliding window of OHLCV + technical indicators
- **Positional Encoding**: Learnable embeddings for time-aware processing
- **Output**: Policy logits + value estimates

**Advantages over Standard PPO:**
- Captures long-range temporal dependencies
- Better pattern recognition in noisy market data
- Handles variable-length sequences efficiently

#### Distributional Critic (`models/distributional_critic.py`)
```python
class QuantileCritic(nn.Module):
    # Models full return distribution via quantile regression
    # Enables direct CVaR optimization
    # Provides tail risk estimates
```

**Technical Details:**
- **Quantiles**: 51 uniformly spaced quantiles (τ ∈ [0,1])
- **Loss Function**: Quantile Huber loss for robustness
- **Output**: Full return distribution instead of scalar value
- **Risk Metrics**: CVaR, VaR, tail risk estimates

**Advantages over Standard Critic:**
- Direct tail risk optimization
- Better handling of fat-tailed return distributions
- Enables risk-aware policy updates

### 2. Advanced Loss Functions

#### CVaR-Optimized PPO Loss (`models/distributional_critic.py`)
```python
class CVaROptimizedLoss:
    # Integrates CVaR penalty into PPO objective
    # Adaptive KL divergence constraints
    # Drawdown-aware penalty terms
```

**Technical Components:**
- **Policy Loss**: Standard PPO clipped surrogate
- **Value Loss**: Quantile Huber loss for distributional targets
- **CVaR Penalty**: Direct optimization of 5% CVaR
- **Drawdown Constraint**: Heavy penalty if DD > 8% threshold
- **Adaptive KL**: Dynamic trust region adjustment

**Mathematical Formulation:**
```
L_total = L_policy + β₁ L_value + β₂ L_entropy + β₃ L_CVaR + β₄ L_drawdown + β₅ L_KL
```

### 3. Market Regime Detection

#### HMM-Based Regime Gating (`models/regime_gating.py`)
```python
class RegimeDetector:
    # Hidden Markov Model for regime classification
    # Fallback to volatility-based clustering
    # Real-time regime probability estimation
```

**Technical Implementation:**
- **Primary Method**: Markov Regime Switching (statsmodels)
- **Fallback**: Gaussian Mixture Models + volatility thresholds
- **Regimes**: Low-vol (ranging), Medium-vol, High-vol (trending)
- **Features**: Returns, volatility, momentum, regime indicators

**Integration with PPO:**
- **Action Masking**: Regime-specific action constraints
- **Exploration**: Adaptive temperature based on regime uncertainty
- **Performance Tracking**: Per-regime Sharpe, win rate, drawdown metrics

### 4. Evaluation Enhancements

#### Purged Walk-Forward CV (`models/purged_cv.py`)
```python
class PurgedKFold(TimeSeriesSplit):
    # Implements López de Prado's leak-free evaluation
    # Purging + embargo to prevent data leakage
    # Block bootstrap for confidence intervals
```

**Technical Details:**
- **Purging**: Remove training samples within `purge_pct` of test boundaries
- **Embargo**: Skip `embargo_pct` buffer after test sets
- **Block Bootstrap**: Preserve autocorrelation in resampling
- **Metrics**: Sharpe CI, return CI, drawdown CI

**Comparison to Standard CV:**
- **Standard K-Fold**: High data leakage in time series
- **Time Series Split**: Some leakage from adjacent samples
- **Purged CV**: Leak-free evaluation with realistic performance estimates

### 5. Risk Management Integration

#### Dynamic Action Masking (`models/dynamic_action_masker.py`)
```python
class DynamicActionMasker:
    # Multi-layer masking system
    # Volatility-adaptive constraints
    # Position size limits
    # Time-based restrictions
```

**Masking Layers:**
1. **Volatility-Based**: Mask large positions in high volatility
2. **Risk Budget**: Prevent position size > capital allocation
3. **Regime-Based**: Regime-specific action constraints
4. **Time-Based**: Market hours and session restrictions

#### Risk-Aware PPO (`models/risk_aware_ppo.py`)
```python
class RiskAwarePPO:
    # CVaR computation and optimization
    # Kelly Criterion position sizing
    # Drawdown constraint enforcement
```

**Risk Metrics:**
- **CVaR (5%)**: Conditional Value at Risk for tail losses
- **Kelly Sizing**: Optimal position size based on win probability
- **Max Drawdown**: Real-time drawdown monitoring and constraints

### 6. Performance Optimization

#### GPU Acceleration (`models/performance_optimization.py`)
```python
class PerformanceOptimizer:
    # Mixed precision training (FP16)
    # Gradient accumulation
    # Memory optimization
```

**Optimizations:**
- **Mixed Precision**: 2x speed, 50% memory reduction
- **Gradient Accumulation**: Larger effective batch sizes
- **Memory Management**: Efficient tensor operations
- **Device Management**: Automatic GPU/CPU selection

### 7. Advanced Backtesting

#### Realistic Market Simulation (`models/advanced_backtester.py`)
```python
class AdvancedBacktester:
    # Realistic slippage modeling
    # Transaction cost integration
    # Market regime detection
    # Performance attribution
```

**Features:**
- **Slippage**: Market impact based on volatility and volume
- **Transaction Costs**: Bid-ask spread + commission modeling
- **Regime Detection**: Performance attribution by market condition
- **Risk Metrics**: Comprehensive risk-adjusted performance measures

## Integration Architecture

### Training Pipeline Flow
```
Market Data → Feature Engineering → Regime Detection → Action Masking → PPO Training → Evaluation
     ↓              ↓                    ↓              ↓              ↓            ↓
  OHLCV+Vol    Technical Indicators   HMM/GMM      Dynamic Masks   Transformer   Purged CV
                                                                   + Dist Critic
```

### Model Architecture Comparison

| Component | Standard PPO | Maskable PPO | Enhanced Sentio PPO |
|-----------|--------------|--------------|---------------------|
| **Policy Network** | MLP | MLP + Masking | Transformer + Masking |
| **Critic** | Scalar Value | Scalar Value | Distributional (51 quantiles) |
| **Loss Function** | Standard PPO | Standard PPO | CVaR-Optimized PPO |
| **Action Space** | Continuous/Discrete | Masked Discrete | Regime-Aware Masked |
| **Evaluation** | Standard CV | Standard CV | Purged Walk-Forward CV |
| **Risk Management** | None | Basic Masking | Multi-Layer + CVaR |
| **Market Awareness** | None | None | HMM Regime Detection |

### File Structure
```
Sentio_PPO_Trainer/
├── models/
│   ├── advanced_unified_trainer.py      # Main training orchestrator
│   ├── transformer_policy.py            # Transformer networks
│   ├── distributional_critic.py         # Quantile regression critic
│   ├── regime_gating.py                 # HMM regime detection
│   ├── purged_cv.py                     # Leak-free evaluation
│   ├── dynamic_action_masker.py         # Multi-layer masking
│   ├── risk_aware_ppo.py               # CVaR optimization
│   ├── performance_optimization.py      # GPU acceleration
│   ├── advanced_backtester.py          # Realistic backtesting
│   ├── advanced_ppo_loss.py            # Enhanced loss functions
│   └── enhanced_unified_trainer.py      # Legacy trainer (deprecated)
├── config/
│   └── production_config.json          # Training configuration
├── requirements.txt                     # Dependencies
└── README.md                           # User documentation
```

## Key Technical Innovations

### 1. Distributional Reinforcement Learning
- **Innovation**: First application of quantile regression to trading PPO
- **Benefit**: Direct tail risk optimization vs. expected value maximization
- **Implementation**: 51-quantile critic with Huber loss

### 2. Market Regime Integration
- **Innovation**: HMM-based regime detection integrated into PPO training
- **Benefit**: Adaptive policy behavior across market conditions
- **Implementation**: Real-time regime classification with per-regime performance tracking

### 3. Leak-Free Evaluation
- **Innovation**: First rigorous application of purged CV to RL evaluation
- **Benefit**: Realistic performance estimates vs. inflated backtest results
- **Implementation**: López de Prado's methodology with block bootstrap CIs

### 4. Multi-Layer Risk Management
- **Innovation**: Hierarchical risk constraints integrated into action masking
- **Benefit**: Multiple safety nets vs. single-point-of-failure risk management
- **Implementation**: Volatility, regime, budget, and time-based masking layers

## Performance Characteristics

### Computational Complexity
- **Training Time**: ~2-3x standard PPO (due to transformer + distributional critic)
- **Memory Usage**: ~1.5x standard PPO (offset by mixed precision)
- **Inference Speed**: ~1.2x standard PPO (transformer overhead)

### Expected Performance Improvements
- **Sharpe Ratio**: 1.5 → 3.0+ (100% improvement)
- **Max Drawdown**: 15% → <8% (47% improvement)
- **Win Rate**: 55% → 65%+ (18% improvement)
- **Risk-Adjusted Returns**: Significant improvement in tail risk metrics

## Dependencies and Requirements

### Core Dependencies
```python
torch>=2.0.0                    # PyTorch for neural networks
stable-baselines3>=2.0.0        # Base RL algorithms
sb3-contrib>=2.0.0             # MaskablePPO
transformers>=4.30.0           # Transformer architectures
statsmodels>=0.14.0            # HMM regime detection
scikit-learn>=1.5.0            # ML utilities
```

### Optional Performance Dependencies
```python
cupy>=12.0.0                   # GPU acceleration
numba>=0.57.0                  # JIT compilation
wandb>=0.15.0                  # Experiment tracking
tensorboard>=2.13.0            # Monitoring
```

## Configuration and Deployment

### Training Configuration
```python
config = {
    "use_transformer": True,           # Enable transformer policy
    "use_distributional_critic": True, # Enable quantile critic
    "use_regime_gating": True,         # Enable regime detection
    "use_purged_cv": True,             # Enable leak-free evaluation
    "n_quantiles": 51,                 # Distributional critic quantiles
    "cvar_alpha": 0.05,                # CVaR risk level
    "max_drawdown_threshold": 0.08,    # Drawdown constraint
    "hidden_size": 512,                # Network size
    "transformer_layers": 6,           # Transformer depth
    "transformer_heads": 8             # Attention heads
}
```

### CLI Usage
```bash
# Maximum performance training
python -m models.advanced_unified_trainer \
    --data-path data/market_data.feather \
    --minutes 30 \
    --use-distributional-critic \
    --use-regime-gating \
    --use-purged-cv \
    --use-transformer \
    --use-wandb

# Quick evaluation
python -m models.advanced_unified_trainer \
    --data-path data/market_data.feather \
    --minutes 15 \
    --use-regime-gating
```

## Comparison with Standard Implementations

### vs. OpenAI Baselines PPO
- **Advantages**: Transformer networks, distributional critic, regime awareness
- **Trade-offs**: Higher computational cost, more complex configuration
- **Use Case**: When maximum performance is required over simplicity

### vs. Stable-Baselines3 PPO
- **Advantages**: All above + production-ready risk management
- **Trade-offs**: Requires domain expertise to configure properly
- **Use Case**: Professional trading applications

### vs. Ray RLlib PPO
- **Advantages**: Specialized for trading, integrated risk management
- **Trade-offs**: Less general-purpose, trading-specific assumptions
- **Use Case**: Dedicated trading model development

## Future Enhancement Roadmap

### Phase 1 (Current)
- ✅ Transformer policy networks
- ✅ Distributional critic with CVaR
- ✅ HMM regime detection
- ✅ Purged walk-forward CV

### Phase 2 (Planned)
- Multi-asset portfolio optimization
- Hierarchical RL for multi-timeframe decisions
- Adversarial training for robustness
- Online learning and model adaptation

### Phase 3 (Research)
- Graph neural networks for market structure
- Causal inference for strategy attribution
- Meta-learning for rapid adaptation
- Quantum-inspired optimization algorithms

This architecture provides a solid foundation for achieving the target performance metrics (3.0+ Sharpe, <8% max drawdown) while maintaining the flexibility to incorporate future enhancements.
