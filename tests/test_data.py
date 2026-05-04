"""Tests for data loading and preprocessing modules."""

import torch
from torch.utils.data import Dataset
import numpy as np

from bathymetry_ml.data import BathymetryDataset


def test_bathymetry_dataset():
    """Test the BathymetryDataset class."""
    # Create dummy data
    dummy_data = torch.randn(100, 7)  # 100 samples, 7 features
    dummy_targets = torch.randn(100)

    dataset = BathymetryDataset(dummy_data, dummy_targets)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 100
    assert dataset.input_dim == 7

    # Test indexing
    data, target = dataset[0]
    assert data.shape == (7,)
    assert isinstance(target, torch.Tensor)


def test_bathymetry_dataset_inference():
    """Test BathymetryDataset in inference mode (no targets)."""
    dummy_data = torch.randn(50, 7)

    dataset = BathymetryDataset(dummy_data)

    assert len(dataset) == 50
    assert dataset.input_dim == 7

    # Test indexing
    data = dataset[0]
    assert data.shape == (7,)
    assert isinstance(data, torch.Tensor)


def test_preprocess_data_ungrouped():
    """Test preprocessing with grouping disabled."""
    import yaml
    from pathlib import Path
    from bathymetry_ml.data import preprocess_data
    from bathymetry_ml import resolve_path

    # Load existing preprocessing config
    config_path = resolve_path("configs/preprocessing.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Disable grouping
    config["preprocessing"]["group_data"] = False

    try:
        train_data, train_targets, pred_data = preprocess_data(config, visualize=False)

        # Check shapes for ungrouped data (should be flat, not grouped)
        assert train_data.ndim == 2, f"Expected 2D train_data, got {train_data.ndim}D"
        assert train_targets.ndim == 1, f"Expected 1D train_targets, got {train_targets.ndim}D"
        assert pred_data.ndim == 2, f"Expected 2D pred_data, got {pred_data.ndim}D"

        # Check feature dimensions
        assert train_data.shape[1] == 7, f"Expected 7 features, got {train_data.shape[1]}"
        assert pred_data.shape[1] == 7, f"Expected 7 features in pred_data, got {pred_data.shape[1]}"

        # Check data consistency
        assert len(train_data) == len(train_targets)
        assert train_data.dtype == torch.float32

    except FileNotFoundError:
        # Skip if data files not available (expected in CI)
        pass


def test_preprocess_data_grouped():
    """Test preprocessing with grouping enabled."""
    import yaml
    from pathlib import Path
    from bathymetry_ml.data import preprocess_data
    from bathymetry_ml import resolve_path

    # Load existing preprocessing config
    config_path = resolve_path("configs/preprocessing.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Enable grouping
    config["preprocessing"]["group_data"] = True
    config["preprocessing"]["group_size"] = 10

    try:
        train_data, train_targets, pred_data = preprocess_data(config, visualize=False)

        # Check shapes for grouped data
        assert train_data.ndim == 3, f"Expected 3D train_data, got {train_data.ndim}D"
        assert train_targets.ndim == 2, f"Expected 2D train_targets, got {train_targets.ndim}D"
        assert pred_data.ndim == 2, f"Expected 2D pred_data, got {pred_data.ndim}D"

        # Check feature dimensions (group_size x n_features)
        assert train_data.shape[2] == 7, f"Expected 7 features per group, got {train_data.shape[2]}"
        assert train_data.shape[1] == 2 ** config["preprocessing"]["group_size"]

        # Check data consistency
        assert len(train_data) == len(train_targets)

    except FileNotFoundError:
        # Skip if data files not available (expected in CI)
        pass
