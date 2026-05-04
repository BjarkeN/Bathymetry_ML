"""Model registry and factory for bathymetry ML models."""

from pathlib import Path
from typing import Union

from .base import BaseModel
from .svdkl import SVDKL
from .dkl import DKL

__all__ = ["BaseModel", "SVDKL", "DKL", "get_model", "MODEL_REGISTRY"]


MODEL_REGISTRY = {
    "svdkl": SVDKL,
    "dkl": DKL,
}


def get_model(
    name: str,
    config_path: Union[str, Path],
    input_dim: int,
    train_x=None,
    train_y=None,
) -> BaseModel:
    """Load model by name from registry.
    
    Args:
        name: Model name (must be in MODEL_REGISTRY)
        config_path: Path to model configuration YAML
        input_dim: Number of input features
        train_x: Training data (optional, required for some models like DKL)
        train_y: Training targets (optional, required for some models like DKL)
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model name not in registry
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[name]

    # Special handling for models that require training data
    if name == "dkl":
        if train_x is None or train_y is None:
            raise ValueError("DKL model requires train_x and train_y")
        return model_class.from_config(config_path, input_dim, train_x, train_y)

    return model_class.from_config(config_path, input_dim)


def register_model(name: str, model_class: type):
    """Register a new model in the registry.
    
    Args:
        name: Model name (for lookup)
        model_class: Model class (must inherit from BaseModel)
    """
    if not issubclass(model_class, BaseModel):
        raise TypeError(f"Model class must inherit from BaseModel, got {model_class}")

    MODEL_REGISTRY[name] = model_class
