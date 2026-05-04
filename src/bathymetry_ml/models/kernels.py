"""Custom GP kernels for bathymetry models."""

import torch
import gpytorch


class GaussMarkovKernel(gpytorch.kernels.Kernel):
    """Gauss-Markov (Matérn 3/2) kernel.
    
    Implements the Gauss-Markov covariance function:
    k(x1, x2) = (1 + |x1 - x2| / ℓ) * exp(-|x1 - x2| / ℓ)
    
    where ℓ is the lengthscale parameter.
    """

    is_stationary = True
    has_lengthscale = True

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params) -> torch.Tensor:
        """Compute Gauss-Markov kernel.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            **params: Additional kernel parameters
            
        Returns:
            Kernel matrix
        """
        # Apply lengthscale
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        
        # Calculate distance between inputs
        diff = self.covar_dist(x1_, x2_, **params)
        
        # Prevent divide by 0 errors
        diff = torch.where(diff == 0, torch.tensor(1e-20, device=diff.device), diff)
        
        # Return Gauss-Markov kernel: (1 + diff) * exp(-diff)
        return (1 + diff) * torch.exp(-diff)
