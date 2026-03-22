"""Core utilities for Edge Vision Applications."""

import logging
import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    return OmegaConf.load(config_path)


def get_device() -> str:
    """Get the best available device for inference.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def format_model_size(model_path: str) -> str:
    """Format model file size in human-readable format.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Formatted size string (e.g., "14.2 MB")
    """
    if not os.path.exists(model_path):
        return "N/A"
    
    size_bytes = os.path.getsize(model_path)
    
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


class PerformanceProfiler:
    """Context manager for profiling inference performance."""
    
    def __init__(self, name: str = "inference"):
        """Initialize profiler.
        
        Args:
            name: Name for the profiling session
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        """Start profiling."""
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and log results."""
        if self.start_time:
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_time.elapsed_time(end_time)
            logging.info(f"{self.name} took {elapsed_ms:.2f} ms")


def validate_input_shape(input_tensor: Union[torch.Tensor, np.ndarray], 
                        expected_shape: tuple) -> bool:
    """Validate input tensor shape for edge constraints.
    
    Args:
        input_tensor: Input tensor to validate
        expected_shape: Expected shape tuple
        
    Returns:
        True if shape is valid, False otherwise
    """
    if isinstance(input_tensor, torch.Tensor):
        actual_shape = input_tensor.shape
    else:
        actual_shape = input_tensor.shape
        
    return actual_shape == expected_shape


def create_sample_data(input_shape: tuple, 
                      data_type: str = "random") -> np.ndarray:
    """Create sample data for testing and demonstration.
    
    Args:
        input_shape: Shape of the input data
        data_type: Type of sample data ('random', 'zeros', 'ones')
        
    Returns:
        Sample data array
    """
    if data_type == "random":
        return np.random.rand(*input_shape).astype(np.float32)
    elif data_type == "zeros":
        return np.zeros(input_shape, dtype=np.float32)
    elif data_type == "ones":
        return np.ones(input_shape, dtype=np.float32)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def mask_sensitive_data(data: np.ndarray, 
                       mask_regions: list) -> np.ndarray:
    """Mask sensitive regions in data for privacy protection.
    
    Args:
        data: Input data array
        mask_regions: List of (x, y, w, h) tuples for regions to mask
        
    Returns:
        Data with masked regions
    """
    masked_data = data.copy()
    
    for x, y, w, h in mask_regions:
        # Ensure coordinates are within bounds
        x = max(0, min(x, data.shape[1] - 1))
        y = max(0, min(y, data.shape[0] - 1))
        w = min(w, data.shape[1] - x)
        h = min(h, data.shape[0] - y)
        
        # Apply Gaussian blur to mask region
        masked_data[y:y+h, x:x+w] = 0.0
        
    return masked_data
