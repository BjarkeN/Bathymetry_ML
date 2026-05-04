"""Reusable feature extractor networks for deep kernel learning."""

import torch
import torch.nn as nn
from typing import List


class DynamicFeatureExtractor(nn.Sequential):
    """Build neural network feature extractor from config parameters.
    
    Constructs sequential network with configurable layer sizes and activation.
    """

    def __init__(self, layer_dims: List[int], activation: str = "relu", dropout: float = 0.0):
        """Initialize feature extractor.
        
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function type ("relu", "tanh", "elu", etc.)
            dropout: Dropout probability (0.0 for no dropout)
        """
        layers = []
        
        # Select activation function
        if activation.lower() == "relu":
            activation_fn = nn.ReLU()
        elif activation.lower() == "tanh":
            activation_fn = nn.Tanh()
        elif activation.lower() == "elu":
            activation_fn = nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build network
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            
            # Add activation and dropout for all but last layer
            if i < len(layer_dims) - 2:
                layers.append(activation_fn)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        
        super().__init__(*layers)


class SVDKLFeatureExtractor(DynamicFeatureExtractor):
    """Feature extractor pre-configured for SVDKL model.
    
    Default architecture: [input_dim, 1024, 1024, 1024, 1024, 1024, 1024, 6]
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """Initialize SVDKL feature extractor.
        
        Args:
            input_dim: Number of input features
            layer_dims: Custom layer dimensions (None uses default)
            activation: Activation function type
            dropout: Dropout probability
        """
        if layer_dims is None:
            layer_dims = [input_dim, 1024, 1024, 1024, 1024, 1024, 1024, 6]
        
        super().__init__(layer_dims, activation, dropout)


class DKLFeatureExtractor(DynamicFeatureExtractor):
    """Feature extractor pre-configured for DKL model.
    
    Default architecture: [input_dim, 1000, 500, 500, 500, 500, 50, 3]
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int] = None,
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """Initialize DKL feature extractor.
        
        Args:
            input_dim: Number of input features
            layer_dims: Custom layer dimensions (None uses default)
            activation: Activation function type
            dropout: Dropout probability
        """
        if layer_dims is None:
            layer_dims = [input_dim, 1000, 500, 500, 500, 500, 50, 3]
        
        super().__init__(layer_dims, activation, dropout)
