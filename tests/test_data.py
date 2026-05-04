"""Tests for data loading and preprocessing modules."""

import torch
from torch.utils.data import Dataset

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
