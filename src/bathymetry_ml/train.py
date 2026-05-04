"""Model training pipeline with HPC support."""

from pathlib import Path
from typing import Dict, Optional, Union
import json
import yaml
import sys

import torch
import gpytorch
import typer
import numpy as np
from tqdm import tqdm

from bathymetry_ml import resolve_path
from .data import preprocess_data, get_data_loaders
from .models import get_model
from .hpc import generate_and_save_job_script

app = typer.Typer(help="Training pipeline for bathymetry models")


def load_yaml(path: Union[str, Path]) -> Dict:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML file (absolute or relative to project root)
        
    Returns:
        Configuration dictionary
    """
    path = resolve_path(str(path))
    with open(path) as f:
        return yaml.safe_load(f)


def save_metrics(metrics: Dict, output_path: Union[str, Path]):
    """Save training metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics to save
        output_path: Path to save metrics file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def train_epoch(
    model: torch.nn.Module,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    mll: gpytorch.mlls.MarginalLogLikelihood,
    config: Dict,
    epoch: int,
) -> tuple:
    """Train for one epoch.
    
    Args:
        model: GP model
        likelihood: Likelihood function
        train_loader: Training data loader
        optimizer: Optimizer
        mll: Marginal log likelihood
        config: Training configuration
        epoch: Current epoch number
        
    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.train()
    likelihood.train()

    losses = []
    rmses = []
    log_every = config.get("training", {}).get("log_every_n_batches", 100)

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(x_batch)
        loss = -mll(output, y_batch)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Calculate RMSE
        with torch.no_grad():
            pred_mean = output.mean.cpu().detach().numpy()
            y_true = y_batch.cpu().detach().numpy()
            rmse = np.sqrt(np.mean((pred_mean - y_true) ** 2))
            rmses.append(rmse)

        if (batch_idx + 1) % log_every == 0:
            avg_loss = np.mean(losses[-log_every:])
            avg_rmse = np.mean(rmses[-log_every:])
            print(f"Epoch {epoch}, Batch {batch_idx + 1}: Loss={avg_loss:.4f}, RMSE={avg_rmse:.4f}")
            sys.stdout.flush()

    return np.mean(losses), {"avg_loss": np.mean(losses), "avg_rmse": np.mean(rmses)}


def validate(
    model: torch.nn.Module,
    likelihood: gpytorch.likelihoods.Likelihood,
    val_loader: torch.utils.data.DataLoader,
    mll: gpytorch.mlls.MarginalLogLikelihood,
) -> Dict:
    """Run validation.
    
    Args:
        model: GP model
        likelihood: Likelihood function
        val_loader: Validation data loader
        mll: Marginal log likelihood
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    likelihood.eval()

    losses = []
    rmses = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            output = model(x_batch)
            loss = -mll(output, y_batch)
            losses.append(loss.item())

            pred_mean = output.mean.cpu().detach().numpy()
            y_true = y_batch.cpu().detach().numpy()
            rmse = np.sqrt(np.mean((pred_mean - y_true) ** 2))
            rmses.append(rmse)

    return {"val_loss": np.mean(losses), "val_rmse": np.mean(rmses)}


@app.command()
def main(
    config: Path = typer.Option(
        "configs/training.yaml",
        help="Path to training configuration YAML",
    ),
    visualize: bool = typer.Option(False, help="Run exploratory visualization before training"),
    hpc: bool = typer.Option(False, help="HPC mode (use HPC paths and settings)"),
    generate_job: bool = typer.Option(False, help="Generate HPC job script instead of training"),
):
    """Train bathymetry model.
    
    Supports:
    - Local training: python -m bathymetry_ml.train
    - With visualization: python -m bathymetry_ml.train --visualize
    - Generate HPC job: python -m bathymetry_ml.train --generate-job
    """

    print("=" * 80)
    print("BATHYMETRY ML - TRAINING PIPELINE")
    print("=" * 80)

    # Resolve config paths relative to project root
    config = resolve_path(str(config))
    
    # Load configurations
    training_config = load_yaml(config)
    preprocessing_config_path = resolve_path(training_config["data"]["preprocessing_config"])
    preprocessing_config = load_yaml(preprocessing_config_path)
    model_config_path = resolve_path(training_config["model"]["config"])

    print(f"\nTraining config: {config}")
    print(f"Preprocessing config: {preprocessing_config_path}")
    print(f"Model config: {model_config_path}")
    print(f"Model: {training_config['model']['name']}")

    # HPC job generation mode
    if generate_job:
        print("\n[HPC MODE] Generating job script...")
        hpc_config_path = resolve_path(
            training_config.get("execution", {}).get("hpc_config", "configs/hpc.yaml")
        )
        command = f"python -m bathymetry_ml.train --config {config}"
        output_path = "job_train.sh"

        script_path = generate_and_save_job_script(str(hpc_config_path), command, output_path)
        print(f"Job script generated: {script_path}")
        print(f"Submit with: bsub < {script_path}")
        return

    # Setup device
    device = training_config.get("execution", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"Device: {device}")

    # Set random seed
    seed = training_config.get("execution", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Optional visualization
    if visualize:
        print("\n[PREPROCESSING] Running exploratory visualization...")
        preprocessing_config["visualization"]["enabled"] = True
        from .data import exploratory_run

        exploratory_run(preprocessing_config)

    # Load and preprocess data
    print("\n[PREPROCESSING] Loading and preprocessing data...")
    train_data, train_targets, pred_data = preprocess_data(preprocessing_config, visualize=visualize)

    # Create data loaders
    print("[DATA] Creating data loaders...")
    train_loader, val_loader = get_data_loaders(train_data, train_targets, training_config, device)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    print("\n[MODEL] Initializing model...")
    model_name = training_config["model"]["name"]
    input_dim = train_data.shape[-1]

    if model_name == "dkl":
        # DKL requires training data
        model = get_model(model_name, model_config_path, input_dim, train_data, train_targets)
    else:
        model = get_model(model_name, model_config_path, input_dim)

    model = model.to(device)
    print(f"Model: {model_name}")

    # Setup optimizer and loss
    print("[TRAINING] Setting up optimizer and loss...")

    model_cfg = load_yaml(model_config_path)
    optimizer_cfg = model_cfg.get("optimizer", {})
    optimizer_type = optimizer_cfg.get("type", "Adam")
    lr = optimizer_cfg.get("lr", 1e-5)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            list(model.model.parameters()) + list(model.likelihood.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            list(model.model.parameters()) + list(model.likelihood.parameters()),
            lr=lr,
        )

    # Use PredictiveLogLikelihood for variational models (SVDKL)
    if model_name == "svdkl":
        from gpytorch.mlls import PredictiveLogLikelihood

        mll = PredictiveLogLikelihood(model.likelihood, model.model, num_data=len(train_loader.dataset))
    else:
        # Use ExactMarginalLogLikelihood for exact models (DKL)
        from gpytorch.mlls import ExactMarginalLogLikelihood

        mll = ExactMarginalLogLikelihood(model.likelihood, model.model)

    # Training loop
    print("\n[TRAINING] Starting training...")
    num_epochs = training_config.get("training", {}).get("num_epochs", 80)

    metrics = {
        "train_losses": [],
        "val_losses": [],
        "train_rmses": [],
        "val_rmses": [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        _, train_metrics = train_epoch(
            model.model,
            model.likelihood,
            train_loader,
            optimizer,
            mll,
            training_config,
            epoch + 1,
        )

        # Validate
        val_metrics = validate(model.model, model.likelihood, val_loader, mll)

        # Log metrics
        metrics["train_losses"].append(train_metrics["avg_loss"])
        metrics["train_rmses"].append(train_metrics["avg_rmse"])
        metrics["val_losses"].append(val_metrics["val_loss"])
        metrics["val_rmses"].append(val_metrics["val_rmse"])

        print(
            f"Val Loss: {val_metrics['val_loss']:.4f}, Val RMSE: {val_metrics['val_rmse']:.4f}"
        )

    # Save model and metrics
    print("\n[SAVING] Saving model and metrics...")
    output_cfg = training_config.get("output", {})
    model_path = resolve_path(output_cfg.get("model_save_path", "models/svdkl_latest.pt"))
    results_dir = resolve_path(output_cfg.get("results_dir", "results/"))

    model.save(model_path)
    print(f"Model saved: {model_path}")

    # Save metrics
    metrics_path = results_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved: {metrics_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    app()
