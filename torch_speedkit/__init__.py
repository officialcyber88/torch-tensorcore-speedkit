"""
torch-tensorcore-speedkit: Auditable PyTorch Tensor Core optimization
"""

__version__ = "0.1.0"

from .config import SpeedConfig
from .speed import SpeedContext, apply_speedups
from .trainer import Trainer

__all__ = ["SpeedConfig", "SpeedContext", "apply_speedups", "Trainer", "__version__"]
