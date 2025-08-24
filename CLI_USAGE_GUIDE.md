# PPO Trainer CLI Usage Guide

## Overview

The enhanced `ppo_trainer.py` now includes a powerful command-line interface (CLI) that gives your engineer precise control over the training process. This follows ML industry best practices and significantly improves experimentation workflow.

## 🚀 **Quick Start Commands**

### **1. Quick Test Run (10 minutes)**
Perfect for testing setup and getting immediate feedback:
```bash
python -m models.ppo_trainer --minutes 10
```

### **2. Episode-Based Training**
Train for a specific number of episodes:
```bash
python -m models.ppo_trainer --episodes 500
```

### **3. Custom Output Location**
Specify where to save the trained model:
```bash
python -m models.ppo_trainer --episodes 1000 --output trained_models/ppo_v2_beta
```

### **4. Full Production Training**
Run the complete training with all advanced features:
```bash
python -m models.ppo_trainer
```

## 📋 **Complete CLI Reference**

### **Training Duration Controls**

| Option | Description | Example |
|--------|-------------|---------|
| `--minutes N` | Maximum training time in minutes | `--minutes 30` |
| `--episodes N` | Number of episodes to train | `--episodes 1000` |

**Note**: `--minutes` overrides `--episodes` if both are specified.

### **Output Controls**

| Option | Description | Example |
|--------|-------------|---------|
| `--output PATH` | Base name for model files | `--output models/ppo_production_v1` |

**Output Files Created**:
- `{PATH}.pth` - The trained PyTorch model
- `{PATH}-metadata.json` - Model metadata and performance metrics

### **Hyperparameter Tuning**

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--lr RATE` | Learning rate | 0.0005 | `--lr 0.001` |
| `--batch-size SIZE` | Training batch size | 512 | `--batch-size 256` |

### **Advanced Training Features**

| Option | Description | Default |
|--------|-------------|---------|
| `--no-curriculum` | Disable curriculum learning | Enabled |
| `--no-ensemble` | Disable multi-agent ensemble | Enabled |
| `--agents N` | Number of agents in ensemble | 4 |

## 🎯 **Common Use Cases**

### **Development & Testing**

```bash
# Quick sanity check (5 minutes)
python -m models.ppo_trainer --minutes 5 --no-ensemble --no-curriculum

# Fast iteration with simplified training
python -m models.ppo_trainer --episodes 100 --no-curriculum --agents 2

# Test different learning rates
python -m models.ppo_trainer --minutes 15 --lr 0.001 --output test_lr_001
python -m models.ppo_trainer --minutes 15 --lr 0.0001 --output test_lr_0001
```

### **Hyperparameter Optimization**

```bash
# Test different batch sizes
python -m models.ppo_trainer --episodes 200 --batch-size 256 --output batch_256
python -m models.ppo_trainer --episodes 200 --batch-size 1024 --output batch_1024

# Compare ensemble sizes
python -m models.ppo_trainer --episodes 300 --agents 2 --output ensemble_2
python -m models.ppo_trainer --episodes 300 --agents 8 --output ensemble_8
```

### **Production Training**

```bash
# Full production model with custom name
python -m models.ppo_trainer --output production_models/ppo_v1_$(date +%Y%m%d)

# Long training run with time limit (4 hours)
python -m models.ppo_trainer --minutes 240 --output long_training_4h

# Conservative training (no experimental features)
python -m models.ppo_trainer --no-curriculum --agents 1 --output conservative_v1
```

## 📊 **Training Output**

### **Console Output Example**
```
🚀 Advanced PPO Trainer for Sentio Trader
==================================================
📋 Training Configuration:
   ⏱️  Time Limit: 10 minutes
   📊 Episodes: 600
   🎯 Learning Rate: 0.0005
   📦 Batch Size: 512
   🎓 Curriculum Learning: True
   🤖 Multi-Agent Ensemble: True (4 agents)
==================================================
🧠 Advanced Features:
   🔄 Prioritized Replay: True
   🔍 Curiosity-Driven Exploration: True
   🏗️  Network Architecture: [512, 512, 256]
   📡 LSTM: True, Attention: True
   📚 Curriculum Stages: 3
      📖 Stage 1: simple (threshold: 0.10%)
      📖 Stage 2: moderate (threshold: 0.50%)
      📖 Stage 3: complex (threshold: 1.00%)

🎯 Target: 10% Monthly Returns through Advanced RL
⚡ Starting training...
```

### **Files Generated**
After training completion:
```
models/
├── ppo_model_20250824_160000.pth          # Trained model
├── ppo_model_20250824_160000-metadata.json # Model metadata
└── ppo_advanced_results_20250824_160000.json # Training results
```

## 🔧 **Advanced Usage Patterns**

### **Automated Training Scripts**

Create bash scripts for automated training:

```bash
#!/bin/bash
# train_multiple_models.sh

# Train models with different configurations
python -m models.ppo_trainer --episodes 500 --lr 0.001 --output models/high_lr_v1
python -m models.ppo_trainer --episodes 500 --lr 0.0001 --output models/low_lr_v1
python -m models.ppo_trainer --episodes 500 --no-curriculum --output models/no_curriculum_v1

echo "All models trained successfully!"
```

### **Grid Search Example**

```bash
#!/bin/bash
# hyperparameter_search.sh

learning_rates=(0.0001 0.0005 0.001)
batch_sizes=(256 512 1024)

for lr in "${learning_rates[@]}"; do
    for batch in "${batch_sizes[@]}"; do
        echo "Training with lr=$lr, batch=$batch"
        python -m models.ppo_trainer \
            --episodes 200 \
            --lr $lr \
            --batch-size $batch \
            --output "grid_search/lr_${lr}_batch_${batch}" \
            --minutes 30
    done
done
```

### **Time-Constrained Training**

```bash
# Train during lunch break (1 hour)
python -m models.ppo_trainer --minutes 60 --output lunch_training

# Overnight training (8 hours)
python -m models.ppo_trainer --minutes 480 --output overnight_training

# Weekend long run (48 hours)
python -m models.ppo_trainer --minutes 2880 --output weekend_marathon
```

## 🎯 **Best Practices**

### **1. Start Small**
Always begin with quick test runs to validate setup:
```bash
python -m models.ppo_trainer --minutes 5
```

### **2. Use Descriptive Output Names**
Include date, configuration, and purpose:
```bash
python -m models.ppo_trainer --output "experiments/ppo_high_lr_20250824"
```

### **3. Document Your Experiments**
Keep a training log:
```bash
echo "$(date): Training with --lr 0.001 --batch-size 256" >> training_log.txt
python -m models.ppo_trainer --lr 0.001 --batch-size 256 --output experiment_001
```

### **4. Monitor Resource Usage**
For long training runs, monitor system resources:
```bash
# Run training in background and log output
nohup python -m models.ppo_trainer --minutes 240 > training.log 2>&1 &
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **PyTorch Not Available**
   ```
   WARNING: PyTorch not available. Advanced training will not function.
   ```
   **Solution**: Install PyTorch: `pip install torch`

2. **Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size: `--batch-size 128`

3. **Training Too Slow**
   ```
   Training taking too long...
   ```
   **Solution**: Disable expensive features:
   ```bash
   python -m models.ppo_trainer --no-ensemble --no-curriculum --episodes 100
   ```

### **Performance Tips**

- Use `--minutes` for time-constrained environments
- Start with `--no-ensemble --no-curriculum` for faster iteration
- Increase `--batch-size` if you have more memory
- Use fewer `--agents` for faster training

## 📈 **Interpreting Results**

The trainer provides comprehensive output including:

- **Episode Progress**: Real-time training progress
- **Performance Metrics**: Reward, estimated monthly returns
- **Model Files**: Ready-to-deploy `.pth` and metadata
- **Training Statistics**: Detailed results in JSON format

**Success Indicators**:
- ✅ Positive final average reward
- ✅ Estimated monthly return > 10%
- ✅ Model files generated successfully
- ✅ No training errors or crashes

---

**This enhanced CLI makes PPO training flexible, efficient, and professional-grade for your ML engineering workflow!** 🚀
