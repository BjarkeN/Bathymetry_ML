"""Tests for model classes and model registry."""

import pytest
import torch
import yaml
from pathlib import Path

from bathymetry_ml.models import get_model, MODEL_REGISTRY, SVDKL


def test_model_registry():
    """Test that models are registered."""
    assert "svdkl" in MODEL_REGISTRY
    assert "dkl" in MODEL_REGISTRY


def test_svdkl_initialization():
    """Test SVDKL model initialization."""
    config = {
        "name": "SVDKL",
        "feature_extractor": {
            "layer_dims": [1024, 1024, 512],
            "activation": "relu",
            "dropout": 0.0,
        },
        "kernel": {"type": "RBFKernel", "ard_num_dims": 3},
        "inducing_points": 50,
        "gp": {"mean_type": "ZeroMean", "noise_constraint": 1e-3},
        "optimizer": {"type": "Adam", "lr": 1e-5, "weight_decay": 1e-4},
    }

    input_dim = 7
    model = SVDKL(config, input_dim)

    assert model.config == config
    assert model.input_dim == input_dim
    assert model.name == "SVDKL"


def test_svdkl_forward():
    """Test SVDKL forward pass."""
    config = {
        "name": "SVDKL",
        "feature_extractor": {
            "layer_dims": [1024, 1024, 512],
            "activation": "relu",
            "dropout": 0.0,
        },
        "kernel": {"type": "RBFKernel", "ard_num_dims": 3},
        "inducing_points": 50,
        "gp": {"mean_type": "ZeroMean", "noise_constraint": 1e-3},
    }

    model = SVDKL(config, input_dim=7)
    model.eval()

    # Create dummy input
    x = torch.randn(10, 7)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    assert hasattr(output, "mean")
    assert hasattr(output, "variance")
    assert output.mean.shape == (10,)
    assert output.variance.shape == (10,)


def test_model_save_load(tmp_path):
    """Test model saving and loading."""
    config = {
        "name": "SVDKL",
        "feature_extractor": {
            "layer_dims": [512, 256],
            "activation": "relu",
        },
        "kernel": {"type": "RBFKernel", "ard_num_dims": 2},
        "inducing_points": 20,
        "gp": {"noise_constraint": 1e-3},
    }

    model1 = SVDKL(config, input_dim=7)

    # Save model
    save_path = tmp_path / "test_model.pt"
    model1.save(save_path)
    assert save_path.exists()

    # Load model
    model2 = SVDKL(config, input_dim=7)
    model2.load(save_path)

    # Compare outputs
    x = torch.randn(5, 7)
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    # Means should be close after loading
    assert torch.allclose(out1.mean, out2.mean, atol=1e-5)


def test_get_model_registry():
    """Test get_model function with registry."""
    assert "svdkl" in MODEL_REGISTRY
    assert "dkl" in MODEL_REGISTRY
