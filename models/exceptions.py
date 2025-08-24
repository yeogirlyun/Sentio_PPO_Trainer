"""
Custom Exception Classes for Sentio PPO Training System
Provides consistent error handling across all modules.
"""

import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class SentioError(Exception):
    """Base exception for Sentio PPO system."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """
        Initialize Sentio error.
        
        Args:
            message: Error message
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        
        # Log the error
        logger.error(f"{self.__class__.__name__}: {message}")
        if details:
            logger.error(f"Error details: {details}")
        if cause:
            logger.error(f"Caused by: {cause}")


class ModelError(SentioError):
    """Base class for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Model loading failed."""
    
    def __init__(self, model_path: str, cause: Optional[Exception] = None):
        message = f"Failed to load model from {model_path}"
        details = {"model_path": model_path}
        super().__init__(message, details, cause)


class ModelSaveError(ModelError):
    """Model saving failed."""
    
    def __init__(self, model_path: str, cause: Optional[Exception] = None):
        message = f"Failed to save model to {model_path}"
        details = {"model_path": model_path}
        super().__init__(message, details, cause)


class ModelArchitectureError(ModelError):
    """Model architecture configuration error."""
    
    def __init__(self, architecture_details: str, cause: Optional[Exception] = None):
        message = f"Model architecture error: {architecture_details}"
        details = {"architecture": architecture_details}
        super().__init__(message, details, cause)


class TrainingError(SentioError):
    """Base class for training-related errors."""
    pass


class TrainingConfigurationError(TrainingError):
    """Training configuration is invalid."""
    
    def __init__(self, config_issue: str, config_value: Any = None):
        message = f"Training configuration error: {config_issue}"
        details = {"config_issue": config_issue, "config_value": config_value}
        super().__init__(message, details)


class TrainingConvergenceError(TrainingError):
    """Training failed to converge."""
    
    def __init__(self, episode: int, loss_value: float, cause: Optional[Exception] = None):
        message = f"Training convergence failure at episode {episode} with loss {loss_value}"
        details = {"episode": episode, "loss": loss_value}
        super().__init__(message, details, cause)


class TrainingMemoryError(TrainingError):
    """Training ran out of memory."""
    
    def __init__(self, batch_size: int, memory_used: float, cause: Optional[Exception] = None):
        message = f"Training memory error with batch size {batch_size}, memory used: {memory_used}GB"
        details = {"batch_size": batch_size, "memory_used_gb": memory_used}
        super().__init__(message, details, cause)


class DataError(SentioError):
    """Base class for data-related errors."""
    pass


class DataLoadError(DataError):
    """Data loading failed."""
    
    def __init__(self, data_path: str, cause: Optional[Exception] = None):
        message = f"Failed to load data from {data_path}"
        details = {"data_path": data_path}
        super().__init__(message, details, cause)


class DataValidationError(DataError):
    """Data validation failed."""
    
    def __init__(self, validation_issue: str, data_shape: Optional[tuple] = None):
        message = f"Data validation error: {validation_issue}"
        details = {"validation_issue": validation_issue}
        if data_shape:
            details["data_shape"] = data_shape
        super().__init__(message, details)


class FeatureEngineeringError(DataError):
    """Feature engineering failed."""
    
    def __init__(self, feature_name: str, cause: Optional[Exception] = None):
        message = f"Feature engineering error for feature: {feature_name}"
        details = {"feature_name": feature_name}
        super().__init__(message, details, cause)


class EnvironmentError(SentioError):
    """Base class for environment-related errors."""
    pass


class EnvironmentInitializationError(EnvironmentError):
    """Environment initialization failed."""
    
    def __init__(self, env_type: str, cause: Optional[Exception] = None):
        message = f"Failed to initialize environment: {env_type}"
        details = {"environment_type": env_type}
        super().__init__(message, details, cause)


class EnvironmentStepError(EnvironmentError):
    """Environment step execution failed."""
    
    def __init__(self, step_number: int, action: Any, cause: Optional[Exception] = None):
        message = f"Environment step {step_number} failed with action {action}"
        details = {"step_number": step_number, "action": action}
        super().__init__(message, details, cause)


class ObservationSpaceError(EnvironmentError):
    """Observation space configuration error."""
    
    def __init__(self, expected_shape: tuple, actual_shape: tuple):
        message = f"Observation space mismatch: expected {expected_shape}, got {actual_shape}"
        details = {"expected_shape": expected_shape, "actual_shape": actual_shape}
        super().__init__(message, details)


class ActionSpaceError(EnvironmentError):
    """Action space configuration error."""
    
    def __init__(self, action_issue: str, action_value: Any = None):
        message = f"Action space error: {action_issue}"
        details = {"action_issue": action_issue, "action_value": action_value}
        super().__init__(message, details)


class DeviceError(SentioError):
    """Base class for device-related errors."""
    pass


class DeviceMismatchError(DeviceError):
    """Tensor device mismatch error."""
    
    def __init__(self, tensor1_device: str, tensor2_device: str):
        message = f"Device mismatch: tensor1 on {tensor1_device}, tensor2 on {tensor2_device}"
        details = {"tensor1_device": tensor1_device, "tensor2_device": tensor2_device}
        super().__init__(message, details)


class GPUMemoryError(DeviceError):
    """GPU memory error."""
    
    def __init__(self, memory_required: float, memory_available: float):
        message = f"GPU memory error: required {memory_required}GB, available {memory_available}GB"
        details = {"memory_required_gb": memory_required, "memory_available_gb": memory_available}
        super().__init__(message, details)


class ConfigurationError(SentioError):
    """Base class for configuration-related errors."""
    pass


class ConfigurationFileError(ConfigurationError):
    """Configuration file error."""
    
    def __init__(self, config_path: str, cause: Optional[Exception] = None):
        message = f"Configuration file error: {config_path}"
        details = {"config_path": config_path}
        super().__init__(message, details, cause)


class ConfigurationValidationError(ConfigurationError):
    """Configuration validation error."""
    
    def __init__(self, parameter_name: str, parameter_value: Any, validation_rule: str):
        message = f"Configuration validation error: {parameter_name}={parameter_value} violates {validation_rule}"
        details = {
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "validation_rule": validation_rule
        }
        super().__init__(message, details)


# Error handling utilities
class ErrorHandler:
    """Utility class for consistent error handling."""
    
    @staticmethod
    def handle_model_error(func):
        """Decorator for handling model-related errors."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                raise ModelLoadError(str(e), cause=e)
            except RuntimeError as e:
                if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                    raise GPUMemoryError(0, 0, cause=e)  # Will be filled by caller
                else:
                    raise ModelError(f"Runtime error in {func.__name__}: {e}", cause=e)
            except Exception as e:
                raise ModelError(f"Unexpected error in {func.__name__}: {e}", cause=e)
        return wrapper
    
    @staticmethod
    def handle_training_error(func):
        """Decorator for handling training-related errors."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                if "shape" in str(e).lower():
                    raise DataValidationError(f"Shape error in {func.__name__}: {e}", cause=e)
                else:
                    raise TrainingConfigurationError(f"Value error in {func.__name__}: {e}")
            except Exception as e:
                import torch
                # Check for CUDA out of memory error
                if hasattr(torch.cuda, 'OutOfMemoryError') and isinstance(e, torch.cuda.OutOfMemoryError):
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    raise TrainingMemoryError(
                        batch_size=kwargs.get('batch_size', 'unknown'),
                        memory_used=allocated,
                        cause=e
                    )
                else:
                    raise TrainingError(f"Unexpected training error in {func.__name__}: {e}", cause=e)
        return wrapper
    
    @staticmethod
    def handle_data_error(func):
        """Decorator for handling data-related errors."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                raise DataLoadError(str(e), cause=e)
            except pd.errors.EmptyDataError as e:
                raise DataValidationError("Empty data file", cause=e)
            except KeyError as e:
                raise DataValidationError(f"Missing required column: {e}", cause=e)
            except Exception as e:
                raise DataError(f"Unexpected data error in {func.__name__}: {e}", cause=e)
        return wrapper


# Context managers for error handling
class ErrorContext:
    """Context manager for consistent error handling."""
    
    def __init__(self, operation_name: str, error_class: type = SentioError):
        self.operation_name = operation_name
        self.error_class = error_class
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Re-raise as our custom error if it's not already
            if not isinstance(exc_val, SentioError):
                raise self.error_class(
                    f"Error in {self.operation_name}: {exc_val}",
                    cause=exc_val
                )
        return False  # Don't suppress the exception


# Validation utilities
def validate_tensor_shape(tensor, expected_shape: tuple, tensor_name: str = "tensor"):
    """Validate tensor shape."""
    if tensor.shape != expected_shape:
        raise DataValidationError(
            f"{tensor_name} shape mismatch: expected {expected_shape}, got {tensor.shape}",
            data_shape=tensor.shape
        )


def validate_config_parameter(param_name: str, param_value: Any, validation_func, validation_desc: str):
    """Validate configuration parameter."""
    if not validation_func(param_value):
        raise ConfigurationValidationError(param_name, param_value, validation_desc)


def validate_device_compatibility(tensor1, tensor2, operation_name: str = "operation"):
    """Validate that tensors are on compatible devices."""
    if tensor1.device != tensor2.device:
        raise DeviceMismatchError(str(tensor1.device), str(tensor2.device))


# Import guard for optional dependencies
def safe_import(module_name: str, error_message: str = None):
    """Safely import optional dependencies."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        if error_message:
            raise ConfigurationError(error_message, cause=e)
        else:
            raise ConfigurationError(f"Required module {module_name} not available", cause=e)
