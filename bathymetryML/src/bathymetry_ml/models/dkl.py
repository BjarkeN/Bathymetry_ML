"""Deep Kernel Learning model with Exact GP."""

from pathlib import Path
from typing import Union, Dict
import yaml

import torch
import gpytorch

from .base import BaseModel
from .kernels import GaussMarkovKernel
from .feature_extractors import DynamicFeatureExtractor


class DKLModel(gpytorch.models.ExactGP):
    """Deep Kernel Learning with Exact GP.
    
    Combines a deep neural network feature extractor with an exact Gaussian Process
    using grid interpolation for efficiency.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        feature_extractor: torch.nn.Module,
        kernel_config: Dict,
        gp_config: Dict,
    ):
        """Initialize DKL model.
        
        Args:
            train_x: Training input data
            train_y: Training target data
            likelihood: Likelihood model
            feature_extractor: Neural network feature extractor
            kernel_config: Dictionary with kernel configuration
            gp_config: Dictionary with GP configuration
        """
        super().__init__(train_x, train_y, likelihood)

        # Set mean module
        mean_type = gp_config.get("mean_type", "ConstantMean")
        if mean_type == "ZeroMean":
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = gpytorch.means.ConstantMean()

        # Get feature extractor output dimension (last layer output)
        with torch.no_grad():
            output_dim = feature_extractor(train_x[:1]).shape[-1]

        # Set kernel with grid interpolation
        base_kernel_type = kernel_config.get("type", "RBFKernel")

        if base_kernel_type == "GaussMarkov":
            base_kernel = GaussMarkovKernel()
        else:
            ard_num_dims = kernel_config.get("ard_num_dims", output_dim)
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)

        grid_size = gp_config.get("grid_size", 100)
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(base_kernel),
            num_dims=output_dim,
            grid_size=grid_size,
        )

        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass through model.
        
        Args:
            x: Input tensor (batch_size, n_features)
            
        Returns:
            MultivariateNormal predictive distribution
        """
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKL(BaseModel):
    """Deep Kernel Learning model wrapper.
    
    High-level interface implementing BaseModel for bathymetry predictions.
    """

    def __init__(self, config: Dict, input_dim: int, train_x: torch.Tensor = None, train_y: torch.Tensor = None):
        """Initialize DKL model from config.
        
        Args:
            config: Configuration dictionary from YAML
            input_dim: Number of input features
            train_x: Training input data (required for exact GP)
            train_y: Training target data (required for exact GP)
        """
        super().__init__(config, input_dim)

        if train_x is None or train_y is None:
            raise ValueError("DKL requires train_x and train_y for exact GP initialization")

        # Initialize feature extractor
        fe_config = config.get("feature_extractor", {})
        layer_dims = [input_dim] + fe_config.get("layer_dims", [1000, 500, 500, 500, 500, 50, 3])
        activation = fe_config.get("activation", "relu")
        dropout = fe_config.get("dropout", 0.0)

        feature_extractor = DynamicFeatureExtractor(layer_dims, activation, dropout)

        # Initialize likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Create model
        self.model = DKLModel(
            train_x,
            train_y,
            self.likelihood,
            feature_extractor,
            config.get("kernel", {}),
            config.get("gp", {}),
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
        """Forward pass - prediction through likelihood.
        
        Args:
            x: Input tensor (batch_size, n_features)
            
        Returns:
            Distribution with mean and variance
        """
        return self.likelihood(self.model(x))

    def save(self, path: Union[str, Path]):
        """Save model and likelihood to disk.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "likelihood_state": self.likelihood.state_dict(),
                "config": self.config,
                "input_dim": self.input_dim,
            },
            path,
        )

    def load(self, path: Union[str, Path]):
        """Load model and likelihood from disk.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.likelihood.load_state_dict(checkpoint["likelihood_state"])

    @classmethod
    def from_config(
        cls, config_path: Union[str, Path], input_dim: int, train_x: torch.Tensor = None, train_y: torch.Tensor = None
    ) -> "DKL":
        """Create DKL model from YAML config.
        
        Args:
            config_path: Path to model configuration YAML
            input_dim: Number of input features
            train_x: Training input data
            train_y: Training target data
            
        Returns:
            Instantiated DKL model
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(config, input_dim, train_x, train_y)
