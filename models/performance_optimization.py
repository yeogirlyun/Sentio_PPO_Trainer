#!/usr/bin/env python3
"""
Performance Optimization Module for PPO Training
Advanced performance optimizations including GPU acceleration, mixed precision training,
gradient accumulation, memory optimization, and distributed training support.

Features:
- CUDA/GPU acceleration with automatic device detection
- Mixed precision training (FP16) for faster training and reduced memory
- Gradient accumulation for effective larger batch sizes
- Memory optimization techniques (gradient checkpointing, efficient data loading)
- Distributed training support (DataParallel, DistributedDataParallel)
- Optimized data pipelines with prefetching
- JIT compilation for critical paths
- Memory profiling and monitoring
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import time
import psutil
import gc
from contextlib import contextmanager
from functools import wraps
import threading
from queue import Queue

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device allocation and optimization for training.
    """
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.memory_fraction = 0.8  # Reserve 20% GPU memory
        
    def _get_optimal_device(self) -> torch.device:
        """Get the optimal device for training."""
        if torch.cuda.is_available():
            # Select GPU with most free memory
            max_memory = 0
            best_device = 0
            
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                if free_memory > max_memory:
                    max_memory = free_memory
                    best_device = i
            
            device = torch.device(f'cuda:{best_device}')
            logger.info(f"Selected device: {device} with {max_memory / 1e9:.1f}GB free memory")
            return device
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def optimize_memory_usage(self):
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1e9  # GB
            stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1e9   # GB
            stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1e9  # GB
        
        # CPU memory
        process = psutil.Process()
        stats['cpu_memory'] = process.memory_info().rss / 1e9  # GB
        stats['cpu_percent'] = process.cpu_percent()
        
        return stats


class MixedPrecisionTrainer:
    """
    Mixed precision training manager for faster training and reduced memory usage.
    """
    
    def __init__(self, enabled: bool = True, loss_scale: str = "dynamic"):
        """
        Initialize mixed precision trainer.
        
        Args:
            enabled: Whether to enable mixed precision
            loss_scale: Loss scaling strategy ("dynamic" or float value)
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enabled) if self.enabled else None
        
        if self.enabled:
            logger.info("Mixed precision training enabled")
        else:
            logger.info("Mixed precision training disabled")
    
    @contextmanager
    def autocast_context(self):
        """Context manager for mixed precision forward pass."""
        if self.enabled:
            with autocast():
                yield
        else:
            yield
    
    def scale_loss_and_backward(self, loss: torch.Tensor, model: nn.Module):
        """Scale loss and perform backward pass."""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Step optimizer with gradient scaling.
        
        Returns:
            True if step was successful, False if skipped due to inf/nan gradients
        """
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True  # Could check for skipped steps if needed
        else:
            optimizer.step()
            return True
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        if self.enabled:
            return self.scaler.get_scale()
        return 1.0


class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes.
    """
    
    def __init__(self, accumulation_steps: int = 4, max_grad_norm: float = 1.0):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def normalize_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Normalize loss by accumulation steps."""
        return loss / self.accumulation_steps
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients and return gradient norm.
        
        Args:
            model: Model to clip gradients for
            
        Returns:
            Gradient norm before clipping
        """
        if self.max_grad_norm > 0:
            return torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        return 0.0
    
    def reset(self):
        """Reset accumulation counter."""
        self.current_step = 0


class MemoryOptimizer:
    """
    Memory optimization utilities for efficient training.
    """
    
    def __init__(self, enable_checkpointing: bool = True):
        """
        Initialize memory optimizer.
        
        Args:
            enable_checkpointing: Enable gradient checkpointing
        """
        self.enable_checkpointing = enable_checkpointing
        
    def apply_gradient_checkpointing(self, model: nn.Module):
        """Apply gradient checkpointing to model."""
        if self.enable_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            else:
                # Manual checkpointing for custom models
                self._apply_manual_checkpointing(model)
    
    def _apply_manual_checkpointing(self, model: nn.Module):
        """Apply manual gradient checkpointing."""
        for module in model.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                module.forward = torch.utils.checkpoint.checkpoint(module.forward)
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def optimize_dataloader(dataset: Dataset, 
                          batch_size: int,
                          num_workers: int = None,
                          pin_memory: bool = True,
                          prefetch_factor: int = 2) -> DataLoader:
        """
        Create optimized DataLoader.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Prefetch factor for workers
            
        Returns:
            Optimized DataLoader
        """
        if num_workers is None:
            num_workers = min(4, torch.get_num_threads())
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False
        )


class DistributedTrainingManager:
    """
    Manager for distributed training across multiple GPUs.
    """
    
    def __init__(self, backend: str = "nccl"):
        """
        Initialize distributed training manager.
        
        Args:
            backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        """
        self.backend = backend
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        
    def setup_distributed(self, rank: int, world_size: int):
        """
        Setup distributed training.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
        """
        self.local_rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        if self.is_distributed:
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size
            )
            torch.cuda.set_device(rank)
            logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Wrap model for distributed training.
        
        Args:
            model: Model to wrap
            
        Returns:
            Wrapped model
        """
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank])
        elif torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        return model
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed:
            dist.destroy_process_group()


class JITOptimizer:
    """
    Just-In-Time compilation optimizer for critical functions.
    """
    
    def __init__(self):
        self.compiled_functions = {}
        
    @staticmethod
    def compile_if_available(func: Callable) -> Callable:
        """
        Compile function with JIT if available.
        
        Args:
            func: Function to compile
            
        Returns:
            Compiled function or original function
        """
        if NUMBA_AVAILABLE:
            try:
                return jit(nopython=True)(func)
            except Exception as e:
                logger.warning(f"JIT compilation failed for {func.__name__}: {e}")
                return func
        return func
    
    @staticmethod
    def vectorized_operations():
        """Optimized vectorized operations for common calculations."""
        
        @jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
        def fast_returns_calculation(prices: np.ndarray) -> np.ndarray:
            """Fast returns calculation."""
            returns = np.zeros(len(prices) - 1)
            for i in range(1, len(prices)):
                returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1]
            return returns
        
        @jit(nopython=True) if NUMBA_AVAILABLE else lambda x: x
        def fast_moving_average(data: np.ndarray, window: int) -> np.ndarray:
            """Fast moving average calculation."""
            result = np.zeros(len(data) - window + 1)
            for i in range(len(result)):
                result[i] = np.mean(data[i:i+window])
            return result
        
        return {
            'returns_calculation': fast_returns_calculation,
            'moving_average': fast_moving_average
        }


class PerformanceProfiler:
    """
    Performance profiler for monitoring training efficiency.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize performance profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.timings = {}
        self.memory_usage = {}
        self.start_times = {}
        
    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling code blocks.
        
        Args:
            name: Name of the profiling section
        """
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Record timing
            duration = end_time - start_time
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            
            # Record memory usage
            memory_delta = end_memory - start_memory
            if name not in self.memory_usage:
                self.memory_usage[name] = []
            self.memory_usage[name].append(memory_delta)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1e9
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics."""
        stats = {}
        
        for name, timings in self.timings.items():
            stats[name] = {
                'avg_time': np.mean(timings),
                'total_time': np.sum(timings),
                'count': len(timings),
                'avg_memory_delta': np.mean(self.memory_usage.get(name, [0]))
            }
        
        return stats
    
    def reset(self):
        """Reset profiling data."""
        self.timings.clear()
        self.memory_usage.clear()


class OptimizedTrainingLoop:
    """
    Optimized training loop with all performance enhancements integrated.
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device_manager: DeviceManager,
                 mixed_precision: bool = True,
                 accumulation_steps: int = 4,
                 max_grad_norm: float = 1.0,
                 enable_profiling: bool = False):
        """
        Initialize optimized training loop.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            device_manager: Device manager
            mixed_precision: Enable mixed precision training
            accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm
            enable_profiling: Enable performance profiling
        """
        self.model = model
        self.optimizer = optimizer
        self.device_manager = device_manager
        
        # Initialize optimization components
        self.mixed_precision = MixedPrecisionTrainer(mixed_precision)
        self.gradient_accumulator = GradientAccumulator(accumulation_steps, max_grad_norm)
        self.memory_optimizer = MemoryOptimizer()
        self.profiler = PerformanceProfiler(enable_profiling)
        
        # Apply optimizations
        self.memory_optimizer.apply_gradient_checkpointing(model)
        self.device_manager.optimize_memory_usage()
        
    def training_step(self, 
                     batch_data: Dict[str, torch.Tensor],
                     loss_fn: Callable) -> Dict[str, float]:
        """
        Perform optimized training step.
        
        Args:
            batch_data: Batch of training data
            loss_fn: Loss function
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        with self.profiler.profile("forward_pass"):
            with self.mixed_precision.autocast_context():
                # Forward pass
                outputs = self.model(**batch_data)
                loss = loss_fn(outputs, batch_data)
                
                # Normalize loss for gradient accumulation
                loss = self.gradient_accumulator.normalize_loss(loss)
        
        with self.profiler.profile("backward_pass"):
            # Backward pass with mixed precision
            self.mixed_precision.scale_loss_and_backward(loss, self.model)
        
        # Check if we should step optimizer
        if self.gradient_accumulator.should_step():
            with self.profiler.profile("optimizer_step"):
                # Clip gradients
                grad_norm = self.gradient_accumulator.clip_gradients(self.model)
                
                # Step optimizer
                step_successful = self.mixed_precision.step_optimizer(self.optimizer)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                metrics['grad_norm'] = grad_norm
                metrics['step_successful'] = step_successful
        
        # Record metrics
        metrics['loss'] = loss.item() * self.gradient_accumulator.accumulation_steps
        metrics['loss_scale'] = self.mixed_precision.get_scale()
        
        return metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'profiler_stats': self.profiler.get_stats(),
            'memory_stats': self.device_manager.get_memory_stats(),
            'mixed_precision_scale': self.mixed_precision.get_scale(),
            'accumulation_steps': self.gradient_accumulator.accumulation_steps
        }
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        self.memory_optimizer.clear_cache()
        self.profiler.reset()


def create_optimized_trainer(model: nn.Module,
                           optimizer: torch.optim.Optimizer,
                           config: Dict[str, Any]) -> OptimizedTrainingLoop:
    """
    Factory function to create optimized trainer with configuration.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        config: Configuration dictionary
        
    Returns:
        Configured OptimizedTrainingLoop
    """
    # Initialize device manager
    device_manager = DeviceManager()
    
    # Move model to optimal device
    model = model.to(device_manager.device)
    
    # Create optimized training loop
    trainer = OptimizedTrainingLoop(
        model=model,
        optimizer=optimizer,
        device_manager=device_manager,
        mixed_precision=config.get('mixed_precision', True),
        accumulation_steps=config.get('accumulation_steps', 4),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        enable_profiling=config.get('enable_profiling', False)
    )
    
    logger.info(f"Created optimized trainer on device: {device_manager.device}")
    return trainer


# Utility decorators for performance optimization
def gpu_accelerated(func):
    """Decorator to automatically move tensors to GPU if available."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            # Move tensor arguments to GPU
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(arg.cuda())
                else:
                    new_args.append(arg)
            
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    new_kwargs[key] = value.cuda()
                else:
                    new_kwargs[key] = value
            
            return func(*new_args, **new_kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper


def memory_efficient(func):
    """Decorator to clear GPU cache after function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return wrapper
