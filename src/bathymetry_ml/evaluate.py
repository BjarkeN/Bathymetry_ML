"""Model evaluation and prediction with HPC support."""

from pathlib import Path
from typing import Dict, Union, Optional
import json
import yaml
import sys

import torch
import gpytorch
import typer
import numpy as np

from bathymetry_ml import resolve_path
from .data import preprocess_data, BathymetryDataset
from .hpc import generate_and_save_job_script

app = typer.Typer(help="Evaluation and prediction pipeline")


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


def load_model(model_path: Union[str, Path]) -> tuple:
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Tuple of (model, likelihood, config)
    """
    checkpoint = torch.load(model_path)

    # Reconstruct model from checkpoint
    from .models import get_model

    config = checkpoint["config"]
    input_dim = checkpoint["input_dim"]
    model_name = config.get("name", "unknown")

    # Create fresh model instance
    if model_name == "svdkl":
        from .models import SVDKL

        model = SVDKL(config, input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Load state
    model.model.load_state_dict(checkpoint["model_state"])
    model.likelihood.load_state_dict(checkpoint["likelihood_state"])

    return model, model.likelihood, config


def predict_batch(
    model: torch.nn.Module,
    likelihood: gpytorch.likelihoods.Likelihood,
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple:
    """Run predictions on data loader.
    
    Args:
        model: Trained model
        likelihood: Likelihood function
        data_loader: DataLoader with prediction data
        device: Device ("cuda" or "cpu")
        
    Returns:
        Tuple of (means, stds)
    """
    model.eval()
    likelihood.eval()

    means = []
    stds = []

    with torch.no_grad():
        for batch_idx, x_batch in enumerate(data_loader):
            if isinstance(x_batch, (list, tuple)):
                x_batch = x_batch[0]

            x_batch = x_batch.to(device)

            # Get predictions
            preds = model(x_batch)

            # Extract mean and variance
            means.append(preds.mean.cpu())
            stds.append(np.sqrt(preds.variance.cpu()))

            if (batch_idx + 1) % 10 == 0:
                print(f"Prediction batch {batch_idx + 1}/{len(data_loader)}")
                sys.stdout.flush()

    return torch.cat(means), torch.cat(stds)


def save_predictions(
    means: torch.Tensor,
    stds: torch.Tensor,
    output_dir: Union[str, Path],
):
    """Save predictions and uncertainties.
    
    Args:
        means: Predicted means
        stds: Predicted standard deviations
        output_dir: Directory to save predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(means, output_dir / "predictions_mean.pt")
    torch.save(stds, output_dir / "predictions_std.pt")

    print(f"Predictions saved to {output_dir}")
    print(f"  Means shape: {means.shape}")
    print(f"  Stds shape: {stds.shape}")


@app.command()
def main(
    config: Path = typer.Option(
        "configs/training.yaml",
        help="Path to training configuration YAML",
    ),
    model_path: Path = typer.Option(..., help="Path to trained model checkpoint"),
    generate_job: bool = typer.Option(False, help="Generate HPC job script instead of evaluating"),
    hpc: bool = typer.Option(False, help="HPC mode"),
):
    """Run predictions on test/validation data.
    
    Supports:
    - Local evaluation: python -m bathymetry_ml.evaluate --model-path models/svdkl.pt
    - Generate HPC job: python -m bathymetry_ml.evaluate --model-path models/svdkl.pt --generate-job
    """

    print("=" * 80)
    print("BATHYMETRY ML - EVALUATION PIPELINE")
    print("=" * 80)

    # Resolve config paths relative to project root
    config = resolve_path(str(config))
    model_path = resolve_path(str(model_path))
    
    # Load configurations
    training_config = load_yaml(config)
    preprocessing_config_path = resolve_path(training_config["data"]["preprocessing_config"])
    preprocessing_config = load_yaml(preprocessing_config_path)

    print(f"\nTraining config: {config}")
    print(f"Model: {model_path}")

    # HPC job generation mode
    if generate_job:
        print("\n[HPC MODE] Generating job script...")
        hpc_config_path = resolve_path(
            training_config.get("execution", {}).get("hpc_config", "configs/hpc.yaml")
        )
        command = f"python -m bathymetry_ml.evaluate --config {config} --model-path {model_path}"
        output_path = "job_eval.sh"

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

    # Load preprocessing data
    print("\n[PREPROCESSING] Loading data...")
    train_data, train_targets, pred_data = preprocess_data(preprocessing_config)

    # Move to device
    if device == "cuda":
        pred_data = pred_data.cuda()

    # Create prediction loader
    data_cfg = training_config.get("data", {})
    pred_batch_size = data_cfg.get("prediction_minibatch_size", 50000)

    pred_dataset = BathymetryDataset(pred_data)
    pred_loader = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=pred_batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"Prediction batches: {len(pred_loader)}")

    # Load model
    print("\n[MODEL] Loading model...")
    model, likelihood, model_config = load_model(model_path)
    model = model.to(device)
    print(f"Model loaded from: {model_path}")

    # Run predictions
    print("\n[EVALUATION] Running predictions...")
    means, stds = predict_batch(model.model, likelihood, pred_loader, device)

    # Save predictions
    print("\n[SAVING] Saving predictions...")
    output_cfg = training_config.get("output", {})
    results_dir = resolve_path(output_cfg.get("results_dir", "results/"))

    save_predictions(means, stds, results_dir)

    # Compute and save basic statistics
    stats = {
        "mean_prediction": float(means.mean()),
        "std_prediction": float(stds.mean()),
        "min_prediction": float(means.min()),
        "max_prediction": float(means.max()),
        "num_predictions": len(means),
    }

    stats_path = results_dir / "prediction_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved: {stats_path}")
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {stats['mean_prediction']:.4f}")
    print(f"  Uncertainty: {stats['std_prediction']:.4f}")
    print(f"  Range: [{stats['min_prediction']:.4f}, {stats['max_prediction']:.4f}]")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    app()
