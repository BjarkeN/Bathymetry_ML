# Bathymetry ML: Deep Learning for Bathymetry Prediction

A structured Machine Learning Operations (MLOps) project for predicting marine bathymetry from gravity data using Deep Kernel Learning with Gaussian Processes. Fully integrated with HPC support for cluster training.

**Status**: ✅ Production-Ready | **Python**: 3.12+ | **License**: MIT

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Usage Examples](#usage-examples)
- [HPC Integration](#hpc-integration)
- [Data Format](#data-format)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

### What This Project Does

Predicts seafloor bathymetry from satellite gravity measurements using advanced machine learning:
- **Input**: Marine gravity data, satellite observations, location features
- **Output**: Bathymetry predictions with uncertainty estimates
- **Models**: SVDKL (Sparse Variational DKL) and DKL (Deep Kernel Learning)
- **Framework**: PyTorch + GPyTorch for Gaussian Process inference

### Key Features

✅ **Config-Driven** - All hyperparameters in YAML files (no hardcoding)  
✅ **Modular Architecture** - Exchangeable models via BaseModel interface  
✅ **HPC-Ready** - Single command to generate and submit LSF jobs  
✅ **Adaptive GPU Selection** - Automatically configures for v100 or v32gb GPUs  
✅ **Reproducible** - Full metrics logging, random seed control, checkpointing  
✅ **Well-Tested** - Comprehensive unit tests for data and models  
✅ **Documented** - Inline docstrings following Google style  

---

## Project Structure

```
Bathymetry_ML/                  # Project root
├── src/bathymetry_ml/              # Main source code
│   ├── models/                     # Model implementations
│   │   ├── base.py                # BaseModel interface
│   │   ├── svdkl.py               # Sparse Variational DKL (recommended)
│   │   ├── dkl.py                 # Deep Kernel Learning with Exact GP
│   │   ├── kernels.py             # Custom Gauss-Markov kernel
│   │   ├── feature_extractors.py  # Configurable neural networks
│   │   └── __init__.py            # Model registry
│   ├── data.py                    # Data loading & preprocessing
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Prediction/evaluation pipeline
│   ├── visualize.py               # Exploratory analysis & plotting
│   ├── hpc.py                     # HPC job generation
│   ├── hpc_utils.py               # LSF utilities
│   ├── api.py                     # FastAPI endpoints (future)
│   └── __init__.py                # Package initialization (includes resolve_path utility)
│
├── scripts/
│   └── generate_hpc_job.py        # CLI tool for HPC job management
│
├── configs/                        # Configuration files (YAML)
│   ├── preprocessing.yaml         # Data loading settings
│   ├── training.yaml              # Training hyperparameters
│   ├── hpc.yaml                   # HPC cluster configuration
│   └── models/
│       ├── svdkl.yaml             # SVDKL model architecture
│       └── dkl.yaml               # DKL model architecture
│
├── data/                          # Data directories
│   ├── raw/                       # External data mount point (symlink)
│   └── processed/                 # Preprocessed tensors
│
├── modelfiles/                    # Trained model checkpoints
│   └── checkpoints/               # Training checkpoints
│
├── results/                       # Training outputs
│   ├── metrics.json               # Training metrics (loss, RMSE)
│   ├── predictions_mean.pt        # Predicted bathymetry
│   └── predictions_std.pt         # Prediction uncertainties
│
├── reports/
│   └── figures/                   # Generated plots
│
├── logs/                          # HPC job logs
│   └── gpu_bathy*.{out,err}       # LSF job output/error
│
├── tests/                         # Unit tests
│   ├── test_data.py              # Data loading tests
│   ├── test_model.py             # Model tests
│   └── __init__.py
│
├── docs/                          # Documentation
│   ├── mkdocs.yml                # MkDocs configuration
│   └── source/
│       └── index.md              # API documentation
│
├── .github/                       # GitHub configuration
│   └── workflows/
│       └── tests.yaml            # CI/CD tests
│
├── pyproject.toml                # Python project metadata
├── requirements.txt              # Production dependencies
├── requirements_dev.txt          # Development dependencies
├── tasks.py                      # Invoke task automation
├── .pre-commit-config.yaml       # Git pre-commit hooks
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA 11.6+ (for GPU support, optional)
- Conda or Python venv
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Bathymetry_ML
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n bathymetry_ml python=3.12
conda activate bathymetry_ml

# OR using venv
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Optional: Install development dependencies for testing and linting:

```bash
pip install -r requirements_dev.txt
```

### Step 4: Verify Installation

```bash
python -c "import bathymetry_ml; import gpytorch; print('✓ Installation successful')"
pytest tests/ -v  # Run unit tests
```

### Path Resolution

**Important**: All paths in the application are resolved relative to the project root (`Bathymetry_ML/`). This means you can run commands from any directory and paths will work correctly:

```bash
# These all work regardless of your current directory
python -m bathymetry_ml.train --config configs/training.yaml
python -m bathymetry_ml.evaluate --model-path modelfiles/svdkl_latest.pt
python scripts/generate_hpc_job.py generate --config configs/training.yaml
```

The `resolve_path()` utility function in `src/bathymetry_ml/__init__.py` handles this automatically.

---

## Quick Start

### 1. Configure Data Path

Edit `configs/preprocessing.yaml`:

```yaml
data_path: /path/to/external/data  # Replace with your data directory
region: global                      # "global" or "malaysia"
aoi: [-180, 180, -90, 90]          # Area of interest [lon_min, lon_max, lat_min, lat_max]
```

### 2. Local Training (Development)

```bash
# Run exploratory analysis with visualization
python -m bathymetry_ml.train --config configs/training.yaml --visualize

# Or just train (silent mode)
python -m bathymetry_ml.train --config configs/training.yaml
```

**Output**: 
- Model saved to `modelfiles/svdkl_latest.pt`
- Metrics saved to `results/metrics.json`
- Plots saved to `reports/figures/`

### 3. Make Predictions

```bash
python -m bathymetry_ml.evaluate \
  --config configs/training.yaml \
  --model-path modelfiles/svdkl_latest.pt
```

**Output**:
- `results/predictions_mean.pt` - Predicted bathymetry
- `results/predictions_std.pt` - Prediction uncertainties
- `results/prediction_stats.json` - Summary statistics

### 4. Submit to HPC (Cluster Training)

```bash
# Generate job script for review
python scripts/generate_hpc_job.py generate \
  --config configs/training.yaml \
  --output job_train.sh

# Review the generated script
cat job_train.sh

# Submit to HPC
bsub < job_train.sh
```

---

## Configuration Guide

All configuration is handled through YAML files in `configs/`. No code changes needed.

### preprocessing.yaml

Controls data loading and preprocessing:

```yaml
data_path: /path/to/data            # Single path to all data files
region: global                       # "global" or "malaysia"
aoi: [-180, 180, -90, 90]           # Area of interest

preprocessing:
  group_size: 10                     # KDTree recursion levels
  n_skip_visualization: 100          # Sample factor for plots

filtering:
  min_dist_to_coast: 0.1             # Minimum distance to coast (grid units)
  ship_data_required: true           # Only keep points with ship data
  grav_on_topo_bounds: 300           # Max absolute gravity value
  sio_bounds: 400                    # Max absolute SIO value

visualization:
  enabled: false                     # Set true for exploratory mode
  save_plots: true                   # Save PNG files
  show_plots: false                  # Display interactively
```

### training.yaml

Controls training and execution:

```yaml
execution:
  mode: local                        # "local" or "hpc"
  device: cuda                       # "cuda" or "cpu"
  seed: 42                           # Random seed for reproducibility

model:
  name: svdkl                        # "svdkl" or "dkl"
  config: configs/models/svdkl.yaml  # Model-specific config

data:
  preprocessing_config: configs/preprocessing.yaml
  train_minibatch_size: 2076         # Training batch size
  prediction_minibatch_size: 50000   # Prediction batch size
  validation_block_size: 10          # Val set size (batches)

training:
  num_epochs: 80                     # Number of training epochs
  log_every_n_batches: 100           # Logging frequency

output:
  model_save_path: modelfiles/svdkl_latest.pt
  checkpoint_dir: modelfiles/checkpoints/
  results_dir: results/
  metrics_format: json               # "json" or "csv"
```

### hpc.yaml

Configures HPC cluster settings (LSF):

```yaml
environment:
  conda_env: bathymetry_ml           # Conda environment name
  cuda_module: cuda/11.6             # CUDA module to load

lsf:
  queue: gpuv100                     # LSF queue
  job_name: bathy_training           # Job name
  num_cores: 4                       # CPU cores
  num_gpus: 1                        # Number of GPUs
  gpu_type: v100                     # "v100" or "v32gb" (32GB variant)
  walltime: "18:00"                  # Max runtime (HH:MM)
  memory: 16GB                       # Memory per node
  email_notifications: false         # Email on completion
  email_address: ""                  # Your email

job_output:
  logs_dir: logs/                    # Where to save job logs
  model_save_path: modelfiles/       # Where to save models
  results_dir: results/              # Where to save results
```

### Model Configs (svdkl.yaml / dkl.yaml)

Define model architecture:

**configs/models/svdkl.yaml**:
```yaml
name: SVDKL
description: Sparse Variational Deep Kernel Learning

feature_extractor:
  input_dim: 7                       # Auto-determined from data
  layer_dims: [1024, 1024, 1024, 1024, 1024, 1024, 6]
  activation: relu                   # "relu", "tanh", "elu"
  dropout: 0.0                       # Dropout probability

kernel:
  type: RBFKernel                    # "RBFKernel" or "GaussMarkov"
  ard_num_dims: 6                    # Feature dimension for kernel

inducing_points: 100                 # Number of inducing points

gp:
  mean_type: ZeroMean                # "ConstantMean" or "ZeroMean"
  noise_constraint: 1e-3             # Lower bound on noise

optimizer:
  type: Adam                         # Optimizer type
  lr: 1e-5                           # Learning rate
  weight_decay: 1e-4                 # L2 regularization

normalization:
  normalize_inputs: true
  normalize_targets: true
  bounds: [-1.0, 1.0]
```

---

## Usage Examples

### Local Development & Testing

```bash
# 1. Exploratory data analysis
python -m bathymetry_ml.train --config configs/training.yaml --visualize

# 2. Train model
python -m bathymetry_ml.train --config configs/training.yaml

# 3. Evaluate
python -m bathymetry_ml.evaluate \
  --config configs/training.yaml \
  --model-path modelfiles/svdkl_latest.pt

# 4. Visualize training metrics
python -m bathymetry_ml.visualize metrics \
  --metrics-path results/metrics.json \
  --output-dir reports/figures/
```

### Customize Model Architecture

Edit `configs/models/svdkl.yaml`:

```yaml
feature_extractor:
  layer_dims: [2048, 1024, 512, 256, 6]  # Deeper network
  activation: relu
  dropout: 0.1                            # Add regularization

optimizer:
  lr: 5e-6                                # Lower learning rate
```

Then retrain:

```bash
python -m bathymetry_ml.train --config configs/training.yaml
```

### Add Custom Model

1. Create `src/bathymetry_ml/models/mymodel.py`:

```python
from .base import BaseModel

class MyModel(BaseModel):
    def forward(self, x):
        # Your implementation
        pass
    
    @classmethod
    def from_config(cls, config_path, input_dim):
        # Load from YAML
        pass
```

2. Register in `src/bathymetry_ml/models/__init__.py`:

```python
from .mymodel import MyModel
register_model("mymodel", MyModel)
```

3. Create `configs/models/mymodel.yaml` with your architecture

4. Update `configs/training.yaml`:

```yaml
model:
  name: mymodel
  config: configs/models/mymodel.yaml
```

### Adjust for Different Hardware

For **limited GPU memory**:

```yaml
# In configs/training.yaml
data:
  train_minibatch_size: 1024    # Reduce batch size

# In configs/hpc.yaml
lsf:
  gpu_type: v32gb               # Request 32GB GPU
  memory: 32GB
```

For **faster training** on large dataset:

```yaml
data:
  train_minibatch_size: 4096    # Increase batch size

training:
  num_epochs: 50                # Fewer epochs
  log_every_n_batches: 50       # Log less frequently
```

---

## HPC Integration

### Workflow Overview

```
Local Development
    ↓
Generate job script (review)
    ↓
Submit to HPC (bsub)
    ↓
Monitor (bjobs, bpeek)
    ↓
Download results (rsync)
```

### Option 1: Generate for Review (Recommended First Time)

```bash
# Generate job script
python scripts/generate_hpc_job.py generate \
  --config configs/training.yaml \
  --output job_train.sh

# Review generated script
cat job_train.sh

# Submit when ready
bsub < job_train.sh
```

### Option 2: Auto-Submit

```bash
# Generate AND submit in one command
python scripts/generate_hpc_job.py submit \
  --config configs/training.yaml \
  --auto-submit

# Output: Job <12345> submitted
```

### Monitor Job

```bash
# Check job status
bjobs 12345

# View live output
bpeek 12345

# After completion, check logs
cat logs/gpu_bathy*.out
```

### Generate Evaluation Job

```bash
python scripts/generate_hpc_job.py create-eval-job \
  --config configs/training.yaml \
  --model-path modelfiles/svdkl.pt \
  --output job_eval.sh

bsub < job_eval.sh
```

### Customize HPC Settings

Edit `configs/hpc.yaml` before submitting:

```yaml
lsf:
  walltime: "24:00"          # Increase time limit
  memory: 32GB               # More memory
  num_gpus: 2                # Multiple GPUs
  gpu_type: v32gb            # Specific GPU type
  email_notifications: true  # Email notifications
  email_address: user@dtu.dk
```

Then submit as usual.

---

## Data Format

### Input Data Files

The `data_path` in `configs/preprocessing.yaml` should contain these **netCDF4 files**:

| File | Description | Dimensions |
|------|-------------|-----------|
| `grav_on_topo.nc` | Gravity on topography | (lat, lon) |
| `topo_low.nc` | Low-resolution topography | (lat, lon) |
| `topo_ship.nc` | Ship-based bathymetry **(targets)** | (lat, lon) |
| `grav_SWOT_01.nc` | SWOT gravity data | (lat, lon) |
| `curv_SWOT_02.nc` | SIO variable grid data | (lat, lon) |

All files must have `lat`, `lon`, and `z` variables.

### Preprocessed Data Output

After preprocessing, the following PyTorch tensors are saved:

```
data/processed/{region}/
├── data.torch              # Feature tensor (n_groups, group_size, 7)
├── target.torch            # Target tensor (n_groups, group_size)
└── prediction.torch        # Prediction features (n_points, 7)
```

### Training Output

```
modelfiles/
├── svdkl_latest.pt         # Latest model checkpoint
└── checkpoints/
    ├── epoch_0.pt
    ├── epoch_10.pt
    └── ...

results/
├── metrics.json            # Training metrics
├── predictions_mean.pt     # Predictions (n_predictions,)
├── predictions_std.pt      # Uncertainties (n_predictions,)
└── prediction_stats.json   # Statistics

logs/
├── gpu_bathy12345.out      # Job output
└── gpu_bathy12345.err      # Job errors

reports/figures/
├── distribution_*.png      # Data distributions
├── training_metrics.png    # Loss & RMSE plots
└── predictions.png         # Prediction plots
```

---

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data.py -v
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=bathymetry_ml --cov-report=html
```

Test coverage includes:
- ✅ Dataset loading and preprocessing
- ✅ Model initialization from config
- ✅ Forward pass validation
- ✅ Model save/load functionality

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'bathymetry_ml'"

```bash
pip install -e .  # Install in editable mode from project root
```

### CUDA Out of Memory

Reduce batch size in `configs/training.yaml`:

```yaml
data:
  train_minibatch_size: 1024  # Was 2076
  prediction_minibatch_size: 25000  # Was 50000
```

Or request a larger GPU:

```yaml
# In configs/hpc.yaml
lsf:
  gpu_type: v32gb  # 32GB GPU
  memory: 32GB
```

### Data Loading Errors

Check `configs/preprocessing.yaml`:

```yaml
data_path: /path/to/data  # Must be correct path
```

Verify all required files exist in data directory:

```bash
ls /path/to/data/
# Should show: grav_on_topo.nc, topo_low.nc, topo_ship.nc, grav_SWOT_01.nc, curv_SWOT_02.nc
```

### HPC Job Submission Fails

1. Check conda environment exists:
   ```bash
   conda activate bathymetry_ml
   ```

2. Verify CUDA module is available:
   ```bash
   module avail cuda/11.6
   ```

3. Check data path is accessible from HPC:
   ```bash
   ssh hpc-cluster
   ls /path/to/data/
   ```

4. Review error log:
   ```bash
   cat logs/gpu_bathy*.err
   ```

### Slow Training

Model is training but slowly:
- Reduce `training.log_every_n_batches` to reduce logging overhead
- Increase batch size if GPU memory allows
- Use SVDKL instead of DKL (more scalable)

---

## Model Comparison

| Aspect | SVDKL | DKL |
|--------|-------|-----|
| **Scalability** | ✅ Excellent (sparse GP) | ⚠️ Limited (exact GP) |
| **Accuracy** | ✅ Good | ⚠️ May be better (exact) |
| **Speed** | ✅ Fast | ⚠️ Slower |
| **Memory** | ✅ ~12GB | ⚠️ Data-dependent |
| **Recommended** | ✅ Large datasets | Smaller datasets |

**Recommendation**: Use SVDKL for production (default)

---

## Code Quality

The project follows MLOps best practices:

- **Code Style**: Ruff formatter (120 char lines)
- **Linting**: Ruff checker
- **Type Hints**: Full type annotations
- **Docstrings**: Google style for all functions/classes
- **Tests**: Pytest with coverage tracking
- **Pre-commit Hooks**: Automated checks before commits

Run code quality tools:

```bash
# Format code
ruff format .

# Lint
ruff check . --fix

# Type checking
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

---

## Performance Notes

### GPU Memory Requirements

| Configuration | VRAM | Batch Size |
|--------------|------|-----------|
| SVDKL (standard) | ~12GB | 2076 |
| SVDKL (optimized) | ~16GB | 4000+ |
| DKL | Variable | <1000 |

### Training Time

Typical training on single v100 GPU:
- 80 epochs with SVDKL: 4-6 hours
- 80 epochs with DKL: 8-12 hours

### Prediction Speed

- SVDKL: ~1000 predictions/sec
- DKL: ~500 predictions/sec

---

## Documentation

### Inline Documentation

All modules have Google-style docstrings:

```python
def function(arg1: str, arg2: int) -> float:
    """Short description.
    
    Longer description of what the function does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
    """
```

### API Documentation

Generate MkDocs documentation:

```bash
cd docs
mkdocs serve
# Visit http://localhost:8000
```

---

## References

### Key Papers & Resources

- **GPyTorch**: https://gpytorch.ai/
- **PyTorch**: https://pytorch.org/
- **Deep Kernel Learning**: https://arxiv.org/abs/1702.08896
- **Variational GP Inference**: https://arxiv.org/abs/1611.02174

### Project Template

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bathymetry_ml_2026,
  author = {Nilsson, Bjarke},
  title = {Bathymetry ML: Deep Learning for Bathymetry Prediction from Marine Gravity Data},
  year = {2026},
  organization = {DTU Space},
  url = {https://github.com/your-repo}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Support & Contributing

### Getting Help

1. Check [Troubleshooting](#troubleshooting) section
2. Review example configs in `configs/`
3. Check test files in `tests/` for usage examples
4. See inline docstrings: `help(function_name)`

### Contributing

To contribute:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and run tests: `pytest tests/`
4. Submit pull request

### Reporting Issues

Please include:
- Python version: `python --version`
- GPU info: `nvidia-smi`
- Error message and traceback
- Minimal reproduction steps
- Relevant config files

---

## Contact

**Author**: Bjarke Nilsson  
**Organization**: DTU Space  
**Email**: [contact info]

---

**Last Updated**: May 2026  
**Version**: 1.0.0  
**Status**: Production-Ready ✅
