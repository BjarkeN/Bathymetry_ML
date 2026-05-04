"""Base class for all bathymetry ML models."""

from abc import ABC, abstractmethod
from pathlib import Path
import torch
import gpytorch
from typing import Union


class BaseModel(torch.nn.Module, ABC):
    """Minimal interface for exchangeable models in bathymetry pipeline.
    
    All models must implement forward pass, save/load, and device transfer.
    Input/output conventions:
    - Input: (batch_size, n_features) tensor
    - Output: gpytorch.distributions.MultivariateNormal with mean and variance
    """

    def __init__(self, config: dict, input_dim: int):
        """Initialize base model.
        
        Args:
            config: Dictionary containing model configuration
            input_dim: Number of input features
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.name = config.get("name", "UnnamedModel")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
        """Forward pass returning predictive distribution.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            gpytorch.distributions.MultivariateNormal with mean and variance
        """
        pass

    def train_mode(self):
        """Set model to training mode."""
        self.train()

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.eval()

    def save(self, path: Union[str, Path]):
        """Save model checkpoint to disk.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Union[str, Path]):
        """Load model checkpoint from disk.
        
        Args:
            path: Path to checkpoint file
        """
        self.load_state_dict(torch.load(path))

    def to(self, device: Union[str, torch.device]) -> "BaseModel":
        """Move model to device.
        
        Args:
            device: Device string ("cuda", "cpu") or torch.device
            
        Returns:
            Self for method chaining
        """
        super().to(device)
        return self

    @classmethod
    @abstractmethod
    def from_config(cls, config_path: Union[str, Path], input_dim: int) -> "BaseModel":
        """Instantiate model from YAML config file.
        
        Args:
            config_path: Path to model configuration YAML
            input_dim: Number of input features
            
        Returns:
            Instantiated model
        """
        pass
