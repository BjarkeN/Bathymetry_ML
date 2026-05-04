"""Sparse Variational Deep Kernel Learning model."""

from pathlib import Path
from typing import Union, Dict
import yaml

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import PredictiveLogLikelihood

from .base import BaseModel
from .kernels import GaussMarkovKernel
from .feature_extractors import DynamicFeatureExtractor


class SVDKLModel(ApproximateGP):
    """Sparse Variational Deep Kernel Learning with GP.
    
    Combines a deep neural network feature extractor with a sparse variational GP
    using inducing points for scalability.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        feature_extractor: torch.nn.Module,
        kernel_config: Dict,
        gp_config: Dict,
    ):
        """Initialize SVDKL model.
        
        Args:
            inducing_points: Inducing point locations (n_inducing, n_features)
            feature_extractor: Neural network feature extractor
            kernel_config: Dictionary with kernel configuration
            gp_config: Dictionary with GP configuration
        """
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()

        # Select and configure kernel
        if kernel_config.get("type") == "GaussMarkov":
            self.covar_module = gpytorch.kernels.ScaleKernel(GaussMarkovKernel())
        else:  # Default RBFKernel
            ard_num_dims = kernel_config.get("ard_num_dims", 6)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
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


class SVDKL(BaseModel):
    """Sparse Variational Deep Kernel Learning model wrapper.
    
    High-level interface implementing BaseModel for bathymetry predictions.
    """

    def __init__(self, config: Dict, input_dim: int):
        """Initialize SVDKL model from config.
        
        Args:
            config: Configuration dictionary from YAML
            input_dim: Number of input features
        """
        super().__init__(config, input_dim)

        # Initialize feature extractor
        fe_config = config.get("feature_extractor", {})
        layer_dims = [input_dim] + fe_config.get("layer_dims", [1024, 1024, 1024, 1024, 1024, 1024, 6])
        activation = fe_config.get("activation", "relu")
        dropout = fe_config.get("dropout", 0.0)

        feature_extractor = DynamicFeatureExtractor(layer_dims, activation, dropout)

        # Initialize inducing points
        n_inducing = config.get("inducing_points", 100)
        self.register_buffer(
            "inducing_points_init",
            torch.randn(n_inducing, layer_dims[-1]),
        )

        # Create model
        self.model = SVDKLModel(
            self.inducing_points_init,
            feature_extractor,
            config.get("kernel", {}),
            config.get("gp", {}),
        )

        # Initialize likelihood
        noise_constraint = config.get("gp", {}).get("noise_constraint", 1e-3)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(noise_constraint)
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
    def from_config(cls, config_path: Union[str, Path], input_dim: int) -> "SVDKL":
        """Create SVDKL model from YAML config.
        
        Args:
            config_path: Path to model configuration YAML
            input_dim: Number of input features
            
        Returns:
            Instantiated SVDKL model
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(config, input_dim)
