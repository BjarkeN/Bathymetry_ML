"""Visualization and exploratory analysis for bathymetry data."""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import typer
import yaml

from .data import preprocess_data, exploratory_run

app = typer.Typer(help="Visualization and exploratory analysis")


def load_yaml(path: Union[str, Path]) -> Dict:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(path) as f:
        return yaml.safe_load(f)


def plot_data_distribution(
    data: np.ndarray,
    feature_names: Optional[list] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """Plot distribution of features.
    
    Args:
        data: Data array (n_samples, n_features)
        feature_names: Names of features
        save_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    if data.ndim > 2:
        # Flatten if needed
        original_shape = data.shape
        data = data.reshape(-1, data.shape[-1])

    n_features = data.shape[1]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    for i in range(n_features):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(data[:, i], bins=100, edgecolor="black")
        axes[0].set_title(f"Distribution of {feature_names[i]}")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")

        # Box plot
        axes[1].boxplot(data[:, i])
        axes[1].set_title(f"Box plot of {feature_names[i]}")
        axes[1].set_ylabel("Value")

        if save_dir:
            save_path = Path(save_dir) / f"distribution_{feature_names[i]}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def plot_training_metrics(
    metrics_path: Union[str, Path],
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """Plot training metrics from JSON file.
    
    Args:
        metrics_path: Path to metrics.json file
        save_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    import json

    with open(metrics_path) as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    if "train_losses" in metrics and "val_losses" in metrics:
        axes[0].plot(metrics["train_losses"], label="Train", linewidth=2)
        axes[0].plot(metrics["val_losses"], label="Val", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # RMSE plot
    if "train_rmses" in metrics and "val_rmses" in metrics:
        axes[1].plot(metrics["train_rmses"], label="Train", linewidth=2)
        axes[1].plot(metrics["val_rmses"], label="Val", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Root Mean Squared Error")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "training_metrics.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions(
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """Plot prediction results.
    
    Args:
        predictions: Predicted values
        ground_truth: Optional ground truth values
        uncertainties: Optional prediction uncertainties
        save_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    if ground_truth is None:
        # Plot predictions only
        plt.figure(figsize=(10, 6))
        plt.hist(predictions, bins=100, edgecolor="black")
        plt.xlabel("Predicted Value")
        plt.ylabel("Count")
        plt.title("Distribution of Predictions")
        plt.grid(True, alpha=0.3)
    else:
        # Plot predictions vs ground truth
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        axes[0].scatter(ground_truth, predictions, alpha=0.5, s=10)
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        axes[0].set_xlabel("Ground Truth")
        axes[0].set_ylabel("Predictions")
        axes[0].set_title("Predictions vs Ground Truth")
        axes[0].grid(True, alpha=0.3)

        # Residuals plot
        residuals = predictions - ground_truth
        axes[1].hist(residuals, bins=100, edgecolor="black")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Residuals")
        axes[1].grid(True, alpha=0.3)

        # Add uncertainties if provided
        if uncertainties is not None:
            axes[1].axvline(uncertainties.mean(), color="r", linestyle="--", linewidth=2, label=f"Mean Uncertainty: {uncertainties.mean():.4f}")
            axes[1].legend()

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / "predictions.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


@app.command()
def exploratory(
    config: Path = typer.Option(
        "configs/preprocessing.yaml",
        help="Path to preprocessing configuration",
    ),
    output_dir: Path = typer.Option(
        "reports/figures/",
        help="Directory to save exploratory plots",
    ),
):
    """Run full preprocessing pipeline with exploratory visualization.
    
    Args:
        config: Path to preprocessing.yaml
        output_dir: Where to save generated plots
    """
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Load config
    preprocessing_config = load_yaml(config)

    # Enable visualization in config
    preprocessing_config["visualization"]["enabled"] = True
    preprocessing_config["visualization"]["save_plots"] = True
    preprocessing_config["visualization"]["show_plots"] = False

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfig: {config}")
    print(f"Output directory: {output_dir}")

    # Run exploratory pipeline
    try:
        exploratory_run(preprocessing_config)
        print(f"\nPlots saved to: {output_dir}")
    except Exception as e:
        print(f"Error during exploratory run: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("=" * 80)


@app.command()
def metrics(
    metrics_path: Path = typer.Option(
        "results/metrics.json",
        help="Path to training metrics JSON file",
    ),
    output_dir: Path = typer.Option(
        "reports/figures/",
        help="Directory to save plots",
    ),
    show: bool = typer.Option(False, help="Display plots interactively"),
):
    """Visualize training metrics.
    
    Args:
        metrics_path: Path to metrics.json
        output_dir: Where to save plots
        show: Whether to display plots
    """
    print("=" * 80)
    print("METRICS VISUALIZATION")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMetrics file: {metrics_path}")
    print(f"Output directory: {output_dir}")

    try:
        plot_training_metrics(metrics_path, save_dir=output_dir, show=show)
        print(f"\nMetrics plots saved to: {output_dir}")
    except Exception as e:
        print(f"Error visualizing metrics: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("METRICS VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    app()
