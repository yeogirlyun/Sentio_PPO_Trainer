"""
Device Utilities for Tensor Consistency
Provides utilities to ensure tensor device consistency across the PPO training system.
"""

import torch
from typing import Union, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device consistency for PyTorch tensors and models.
    """
    
    def __init__(self, device: Union[str, torch.device] = None):
        """
        Initialize device manager.
        
        Args:
            device: Target device. If None, auto-detects best available device.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"DeviceManager initialized with device: {self.device}")
        
        # Log GPU info if available
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device.index or 0)
            gpu_memory = torch.cuda.get_device_properties(self.device.index or 0).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    def to_device(self, *tensors: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Move tensors to the target device.
        
        Args:
            *tensors: Variable number of tensors to move
            
        Returns:
            Single tensor or tuple of tensors on target device
        """
        moved_tensors = []
        
        for tensor in tensors:
            if tensor is not None:
                if not isinstance(tensor, torch.Tensor):
                    logger.warning(f"Expected torch.Tensor, got {type(tensor)}. Converting...")
                    tensor = torch.tensor(tensor)
                
                moved_tensors.append(tensor.to(self.device))
            else:
                moved_tensors.append(None)
        
        if len(moved_tensors) == 1:
            return moved_tensors[0]
        return tuple(moved_tensors)
    
    def ensure_same_device(self, *tensors: torch.Tensor) -> bool:
        """
        Check if all tensors are on the same device.
        
        Args:
            *tensors: Tensors to check
            
        Returns:
            True if all tensors are on the same device
        """
        if not tensors:
            return True
        
        # Filter out None tensors
        valid_tensors = [t for t in tensors if t is not None and isinstance(t, torch.Tensor)]
        
        if not valid_tensors:
            return True
        
        first_device = valid_tensors[0].device
        
        for i, tensor in enumerate(valid_tensors[1:], 1):
            if tensor.device != first_device:
                logger.warning(f"Device mismatch: tensor 0 on {first_device}, tensor {i} on {tensor.device}")
                return False
        
        return True
    
    def move_to_device(self, obj: Any) -> Any:
        """
        Recursively move PyTorch objects to target device.
        
        Args:
            obj: Object to move (tensor, model, list, dict, etc.)
            
        Returns:
            Object moved to target device
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, torch.nn.Module):
            return obj.to(self.device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.move_to_device(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self.move_to_device(value) for key, value in obj.items()}
        else:
            return obj
    
    def get_memory_info(self) -> dict:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory information
        """
        info = {'device': str(self.device)}
        
        if self.device.type == 'cuda':
            info.update({
                'allocated_gb': torch.cuda.memory_allocated(self.device) / 1e9,
                'reserved_gb': torch.cuda.memory_reserved(self.device) / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1e9,
                'total_gb': torch.cuda.get_device_properties(self.device).total_memory / 1e9
            })
        else:
            info['type'] = 'cpu'
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")


def ensure_tensor_device_consistency(*tensors: torch.Tensor, target_device: torch.device = None) -> Tuple[torch.Tensor, ...]:
    """
    Ensure all tensors are on the same device.
    
    Args:
        *tensors: Tensors to check and move
        target_device: Target device. If None, uses device of first tensor.
        
    Returns:
        Tuple of tensors all on the same device
    """
    if not tensors:
        return ()
    
    # Filter out None tensors
    valid_tensors = [(i, t) for i, t in enumerate(tensors) if t is not None]
    
    if not valid_tensors:
        return tensors
    
    # Determine target device
    if target_device is None:
        target_device = valid_tensors[0][1].device
    
    # Move all tensors to target device
    result = list(tensors)
    
    for i, tensor in valid_tensors:
        if tensor.device != target_device:
            result[i] = tensor.to(target_device)
    
    return tuple(result)


def safe_tensor_operation(operation_func, *tensors: torch.Tensor, device: torch.device = None):
    """
    Safely perform tensor operations with device consistency checks.
    
    Args:
        operation_func: Function to perform on tensors
        *tensors: Input tensors
        device: Target device for operation
        
    Returns:
        Result of operation with proper device handling
    """
    try:
        # Ensure device consistency
        if device is not None:
            tensors = tuple(t.to(device) if t is not None else t for t in tensors)
        else:
            tensors = ensure_tensor_device_consistency(*tensors)
        
        # Perform operation
        result = operation_func(*tensors)
        
        return result
        
    except RuntimeError as e:
        if "device" in str(e).lower():
            logger.error(f"Device-related error in tensor operation: {e}")
            # Try to fix by moving all tensors to CPU
            cpu_tensors = tuple(t.cpu() if t is not None else t for t in tensors)
            logger.warning("Falling back to CPU computation")
            return operation_func(*cpu_tensors)
        else:
            raise


class DeviceAwareMixin:
    """
    Mixin class to add device awareness to PyTorch modules.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_manager = None
    
    def setup_device_manager(self, device: Union[str, torch.device] = None):
        """Setup device manager for this module."""
        self.device_manager = DeviceManager(device)
        return self
    
    @property
    def device(self) -> torch.device:
        """Get current device."""
        if self.device_manager:
            return self.device_manager.device
        elif hasattr(self, 'parameters'):
            # Try to get device from model parameters
            try:
                return next(self.parameters()).device
            except StopIteration:
                pass
        return torch.device('cpu')
    
    def to_device(self, *tensors: torch.Tensor):
        """Move tensors to module's device."""
        if self.device_manager:
            return self.device_manager.to_device(*tensors)
        else:
            device = self.device
            if len(tensors) == 1:
                return tensors[0].to(device)
            return tuple(t.to(device) for t in tensors)
    
    def ensure_input_device(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is on correct device."""
        target_device = self.device
        if x.device != target_device:
            logger.debug(f"Moving input from {x.device} to {target_device}")
            x = x.to(target_device)
        return x


# Global device manager instance
_global_device_manager = None


def get_global_device_manager() -> DeviceManager:
    """Get or create global device manager."""
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def set_global_device(device: Union[str, torch.device]):
    """Set global device for all operations."""
    global _global_device_manager
    _global_device_manager = DeviceManager(device)
    logger.info(f"Global device set to: {_global_device_manager.device}")


# Convenience functions
def to_device(*tensors: torch.Tensor):
    """Move tensors to global device."""
    return get_global_device_manager().to_device(*tensors)


def get_device() -> torch.device:
    """Get global device."""
    return get_global_device_manager().device


def clear_gpu_cache():
    """Clear GPU cache."""
    get_global_device_manager().clear_cache()


def get_memory_info() -> dict:
    """Get memory information."""
    return get_global_device_manager().get_memory_info()
