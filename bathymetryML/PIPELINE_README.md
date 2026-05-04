# Bathymetry ML Pipeline

A structured machine learning project for predicting bathymetry from marine gravity data using Deep Kernel Learning with Gaussian Processes.

## Project Structure

```
bathymetryML/
├── src/bathymetry_ml/           # Main source code
│   ├── models/                  # Model implementations
│   │   ├── base.py             # BaseModel interface
│   │   ├── svdkl.py            # Sparse Variational DKL
│   │   ├── dkl.py              # Deep Kernel Learning
│   │   ├── kernels.py          # Custom GP kernels
│   │   └── feature_extractors.py
│   ├── data.py                 # Data loading & preprocessing
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Prediction/evaluation
│   ├── visualize.py            # Exploratory analysis
│   ├── hpc.py                  # HPC job generation
│   └── hpc_utils.py            # LSF utilities
├── scripts/
│   └── generate_hpc_job.py     # HPC CLI tool
├── configs/                    # Configuration files
│   ├── preprocessing.yaml      # Data settings
│   ├── training.yaml           # Training settings
│   ├── hpc.yaml               # HPC cluster settings
│   └── models/
│       ├── svdkl.yaml         # SVDKL model config
│       └── dkl.yaml           # DKL model config
├── data/
│   ├── raw/                   # External data (mount point)
│   └── processed/             # Generated tensors
├── models/                    # Trained model checkpoints
├── results/                   # Training outputs
├── reports/
│   └── figures/              # Generated plots
└── tests/                    # Unit tests
```

## Configuration

All settings are configured via YAML files in `configs/`.

### preprocessing.yaml
Data loading and preprocessing settings:
- `data_path`: Path to external data folder
- `region`: "global" or "malaysia"
- `aoi`: Area of interest [lon_min, lon_max, lat_min, lat_max]
- `preprocessing.group_size`: KDTree grouping levels
- `filtering`: Data filtering thresholds

### training.yaml
Training pipeline settings:
- `model.name`: "svdkl" or "dkl"
- `training.num_epochs`: Number of training epochs
- `data.train_minibatch_size`: Batch size
- `execution.device`: "cuda" or "cpu"
- `execution.hpc_config`: Path to HPC config

### hpc.yaml
HPC cluster settings:
- `lsf.gpu_type`: "v100" or "v32gb"
- `lsf.walltime`: Maximum runtime (HH:MM)
- `lsf.memory`: Memory per node
- `environment.conda_env`: Conda environment name
- `environment.cuda_module`: CUDA module to load

### configs/models/
Model-specific configurations (architecture, optimizer, etc.)

## Quick Start

### Local Training

1. **Set data path** in `configs/preprocessing.yaml`:
   ```yaml
   data_path: /path/to/external/data
   ```

2. **Run exploratory analysis** (optional):
   ```bash
   python -m bathymetry_ml.train --config configs/training.yaml --visualize
   ```

3. **Train model**:
   ```bash
   python -m bathymetry_ml.train --config configs/training.yaml
   ```

4. **Evaluate on test data**:
   ```bash
   python -m bathymetry_ml.evaluate \
     --config configs/training.yaml \
     --model-path models/svdkl_latest.pt
   ```

### HPC Workflow

#### Option 1: Generate job script for review (recommended for first run)

```bash
python scripts/generate_hpc_job.py generate \
  --config configs/training.yaml \
  --output job_train.sh

# Review job_train.sh, then submit:
bsub < job_train.sh
```

#### Option 2: Auto-submit directly

```bash
python scripts/generate_hpc_job.py submit \
  --config configs/training.yaml \
  --auto-submit
```

#### Check job status

```bash
# On HPC cluster
bjobs <job_id>
bpeek <job_id>  # View running logs
```

#### Evaluate on HPC

```bash
python scripts/generate_hpc_job.py create-eval-job \
  --config configs/training.yaml \
  --model-path models/svdkl.pt \
  --output job_eval.sh

bsub < job_eval.sh
```

## Usage Examples

### Change Model Architecture

Edit `configs/models/svdkl.yaml`:
```yaml
feature_extractor:
  layer_dims: [2048, 1024, 512, 256, 6]  # Deeper network
  activation: relu
  dropout: 0.1                            # Add dropout
```

### Add Custom Model

1. Create `src/bathymetry_ml/models/mymodel.py` inheriting from `BaseModel`
2. Implement required methods: `forward()`, `from_config()`, `save()`, `load()`
3. Register in `src/bathymetry_ml/models/__init__.py`:
   ```python
   from .mymodel import MyModel
   register_model("mymodel", MyModel)
   ```
4. Create config `configs/models/mymodel.yaml`
5. Update training config: `model.name: mymodel`

### Adjust HPC Settings

Edit `configs/hpc.yaml`:
```yaml
lsf:
  num_gpus: 2              # Request multiple GPUs
  walltime: "24:00"        # Increase time limit
  memory: 32GB             # More memory
  gpu_type: v32gb          # Request 32GB GPU
```

### Visualization

```bash
# Exploratory data analysis
python -m bathymetry_ml.visualize exploratory \
  --config configs/preprocessing.yaml \
  --output-dir reports/figures/

# Plot training metrics
python -m bathymetry_ml.visualize metrics \
  --metrics-path results/metrics.json \
  --output-dir reports/figures/
```

## Training Monitoring

Training metrics are saved to `results/metrics.json`:
- `train_losses`: Training loss per epoch
- `val_losses`: Validation loss per epoch
- `train_rmses`: Training RMSE per epoch
- `val_rmses`: Validation RMSE per epoch

HPC job logs are saved to `logs/`:
- `logs/gpu_bathy*.out`: Standard output
- `logs/gpu_bathy*.err`: Standard error

## Testing

Run tests with pytest:
```bash
pytest tests/
pytest tests/test_data.py -v
pytest tests/test_model.py -v
```

## Data Format Requirements

### External Data Path
The `data_path` in `preprocessing.yaml` should contain:
- `grav_on_topo.nc`: Gravity on topography (netCDF4)
- `topo_low.nc`: Low-resolution topography
- `topo_ship.nc`: Ship-based bathymetry (targets)
- `grav_SWOT_01.nc`: SWOT gravity data
- `curv_SWOT_02.nc`: SIO variable grid data

All files must have `lat`, `lon`, and `z` dimensions.

### Preprocessing Output
Processed data is saved as PyTorch tensors:
- `data/processed/global/data.torch`: Feature tensor (n_groups, group_size, n_features)
- `data/processed/global/target.torch`: Target tensor (ship bathymetry)
- `data/processed/global/prediction.torch`: Prediction data

## Model Outputs

### Training
- `models/svdkl_latest.pt`: Trained model checkpoint
- `results/metrics.json`: Training metrics
- `models/checkpoints/`: Intermediate checkpoints

### Prediction
- `results/predictions_mean.pt`: Predicted bathymetry
- `results/predictions_std.pt`: Prediction uncertainties
- `results/prediction_stats.json`: Summary statistics

### Visualization
- `reports/figures/*.png`: Generated plots

## Performance Notes

- **SVDKL**: Scalable to large datasets, uses sparse variational inference (recommended)
- **DKL**: Exact GP inference, slower but potentially more accurate for smaller datasets
- GPU VRAM requirements (~16GB):
  - SVDKL: ~12GB with batch_size=2076
  - DKL: Highly dependent on dataset size

## Troubleshooting

### CUDA out of memory
- Reduce `data.train_minibatch_size` in `training.yaml`
- Reduce feature extractor `layer_dims` in model config
- Use `v32gb` GPU type in `hpc.yaml`

### HPC job fails
- Check `logs/gpu_bathy*.err` for error messages
- Verify data path is accessible on HPC
- Check conda environment exists on HPC
- Ensure CUDA module version matches PyTorch

### Data loading errors
- Verify `data_path` in `preprocessing.yaml` points to correct directory
- Ensure all required netCDF4 files are present
- Check netCDF file dimensions match expected format

## References

- **GPyTorch**: https://gpytorch.ai/
- **PyTorch**: https://pytorch.org/
- **MkDocs**: `uv run mkdocs serve` (build documentation locally)

## License

MIT License - See LICENSE file for details

## Citation

If you use this code, please cite:

```bibtex
@software{bathymetry_ml_2026,
  author = {Nilsson, Bjarke},
  title = {Bathymetry ML: Deep Learning for Bathymetry Prediction},
  year = {2026},
  organization = {DTU Space}
}
```
