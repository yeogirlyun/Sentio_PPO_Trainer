#!/usr/bin/env python3
"""
Transformer-Based Policy Network for PPO
Advanced neural architecture using multi-head attention for market data processing.

Features:
- Multi-head attention mechanisms for temporal pattern recognition
- Positional encoding for time series data
- Residual connections and layer normalization
- Scheduled dropout for regularization
- Hierarchical feature processing
- Market-specific attention patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from .device_utils import DeviceAwareMixin, ensure_tensor_device_consistency

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series data with learnable components.
    Combines sinusoidal encoding with learnable parameters for market data.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
        # Learnable positional embedding for market-specific patterns
        self.learnable_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1) + self.learnable_pe[:, :seq_len, :]
        return self.dropout(x)


class MarketAttentionHead(nn.Module):
    """
    Specialized attention head for market data with temporal and feature-wise attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Market-specific attention biases
        self.temporal_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.feature_bias = nn.Parameter(torch.zeros(1, num_heads, 1, d_model))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with market-specific biases
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores + self.temporal_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network with market-specific activations.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Market-specific gating mechanism
        self.gate = nn.Linear(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated feed-forward with market-specific patterns
        gate_values = torch.sigmoid(self.gate(x))
        ff_output = self.linear1(x)
        ff_output = self.activation(ff_output) * gate_values
        ff_output = self.dropout(ff_output)
        return self.linear2(ff_output)


class TransformerEncoderLayer(nn.Module):
    """
    Enhanced transformer encoder layer with market-specific modifications.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MarketAttentionHead(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class TransformerPolicyNetwork(DeviceAwareMixin, nn.Module):
    """
    Transformer-based policy network for PPO with market-specific enhancements.
    
    Features:
    - Multi-head attention for temporal pattern recognition
    - Positional encoding for time series data
    - Hierarchical feature processing
    - Separate actor and critic heads
    - Market regime adaptation
    """
    
    def __init__(self, 
                 input_size: int = 2340,
                 d_model: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 action_size: int = 3,
                 max_seq_len: int = 1000,
                 use_market_embedding: bool = True):
        """
        Initialize the Transformer Policy Network.
        
        Args:
            input_size: Size of input features
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            action_size: Number of actions
            max_seq_len: Maximum sequence length
            use_market_embedding: Whether to use market-specific embeddings
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        self.action_size = action_size
        self.use_market_embedding = use_market_embedding
        
        # Input embedding and projection
        self.input_embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Market-specific feature embeddings
        if use_market_embedding:
            self.price_embedding = nn.Linear(5, d_model // 4)  # OHLCV
            self.volume_embedding = nn.Linear(1, d_model // 8)
            self.technical_embedding = nn.Linear(input_size - 6, d_model - d_model // 4 - d_model // 8)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Market regime detection head
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)  # 4 market regimes
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, action_size)
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        
        # Attention pooling for sequence aggregation
        self.attention_pool = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for autoregressive modeling."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the transformer policy network.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size] or [batch_size, input_size]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (policy_logits, value, attention_info)
        """
        # Ensure input is on correct device
        x = self.ensure_input_device(x)
        
        batch_size = x.size(0)
        
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        seq_len = x.size(1)
        device = self.device
        
        # Market-specific feature embedding
        if self.use_market_embedding and x.size(-1) >= 6:
            price_features = self.price_embedding(x[:, :, :5])  # OHLCV
            volume_features = self.volume_embedding(x[:, :, 5:6])  # Volume
            if x.size(-1) > 6:
                tech_features = self.technical_embedding(x[:, :, 6:])  # Technical indicators
                embedded = torch.cat([price_features, volume_features, tech_features], dim=-1)
            else:
                embedded = torch.cat([price_features, volume_features], dim=-1)
                # Pad to d_model if needed
                if embedded.size(-1) < self.d_model:
                    padding = torch.zeros(batch_size, seq_len, self.d_model - embedded.size(-1), device=device)
                    embedded = torch.cat([embedded, padding], dim=-1)
        else:
            embedded = self.input_embedding(x)
        
        # Add positional encoding
        embedded = self.positional_encoding(embedded)
        
        # Create attention mask for causal modeling
        attention_mask = self.create_attention_mask(seq_len, device)
        
        # Pass through transformer layers
        hidden_states = embedded
        attention_weights = []
        
        for layer in self.transformer_layers:
            hidden_states, layer_attention = layer(hidden_states, attention_mask)
            if return_attention:
                attention_weights.append(layer_attention)
        
        # Market regime detection
        regime_logits = self.regime_detector(hidden_states.mean(dim=1))
        
        # Attention pooling for final representation
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        pooled_output, pool_attention = self.attention_pool(
            pool_query, hidden_states, hidden_states
        )
        pooled_output = pooled_output.squeeze(1)  # Remove query dimension
        
        # Actor and critic outputs
        policy_logits = self.actor_head(pooled_output)
        value = self.critic_head(pooled_output)
        
        # Prepare attention information
        attention_info = None
        if return_attention:
            attention_info = {
                'layer_attentions': attention_weights,
                'pool_attention': pool_attention,
                'regime_logits': regime_logits
            }
        
        return policy_logits, value.squeeze(-1), attention_info
    
    def get_action_distribution(self, x: torch.Tensor) -> torch.distributions.Categorical:
        """Get action distribution for sampling."""
        policy_logits, _, _ = self.forward(x)
        return torch.distributions.Categorical(logits=policy_logits)
    
    def get_log_probs(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for given actions."""
        policy_logits, _, _ = self.forward(x)
        dist = torch.distributions.Categorical(logits=policy_logits)
        return dist.log_prob(actions)
    
    def get_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Get policy entropy."""
        policy_logits, _, _ = self.forward(x)
        dist = torch.distributions.Categorical(logits=policy_logits)
        return dist.entropy()
    
    def analyze_attention_patterns(self, x: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns for interpretability."""
        with torch.no_grad():
            _, _, attention_info = self.forward(x, return_attention=True)
            
            if attention_info is None:
                return {}
            
            # Analyze temporal attention patterns
            layer_attentions = attention_info['layer_attentions']
            temporal_focus = []
            
            for layer_attn in layer_attentions:
                # Average across heads and batch
                avg_attn = layer_attn.mean(dim=(0, 1))  # [seq_len, seq_len]
                temporal_focus.append(avg_attn.diagonal().mean().item())
            
            return {
                'temporal_focus_by_layer': temporal_focus,
                'regime_predictions': F.softmax(attention_info['regime_logits'], dim=-1),
                'attention_entropy': [torch.distributions.Categorical(probs=attn.mean(dim=(0, 1))).entropy().item() 
                                    for attn in layer_attentions]
            }


class ScheduledDropout(nn.Module):
    """
    Dropout with scheduled rate reduction during training.
    """
    
    def __init__(self, initial_rate: float = 0.1, final_rate: float = 0.05, total_steps: int = 100000):
        super().__init__()
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_steps = total_steps
        self.current_step = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Linear schedule from initial to final rate
            progress = min(self.current_step / self.total_steps, 1.0)
            current_rate = self.initial_rate + (self.final_rate - self.initial_rate) * progress
            return F.dropout(x, p=current_rate, training=True)
        return x
    
    def step(self):
        """Update the current step for scheduling."""
        self.current_step += 1
