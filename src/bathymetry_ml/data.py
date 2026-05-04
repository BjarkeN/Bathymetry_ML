"""Data loading, preprocessing, and dataset classes for bathymetry ML."""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
import scipy.interpolate
import torch
import netCDF4 as nc
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import yaml

from bathymetry_ml import resolve_path


def load_netcdf_data(data_path: Union[str, Path]) -> Dict:
    """Load netCDF4 files from data folder.
    
    Args:
        data_path: Path to folder containing netCDF files
        
    Returns:
        Dictionary with loaded datasets
    """
    data_path = Path(data_path)

    datasets = {
        "grav_on_topo": nc.Dataset(data_path / "grav_on_topo.nc", "r"),
        "topo_low": nc.Dataset(data_path / "topo_low.nc", "r"),
        "topo_ship": nc.Dataset(data_path / "topo_ship.nc", "r"),
        "grav_SWOT": nc.Dataset(data_path / "grav_SWOT_01.nc", "r"),
        "sio_vgg": nc.Dataset(data_path / "curv_SWOT_03.nc", "r"),
    }

    return datasets


def slice_data(data: nc.Dataset, aoi: list, datakey: str = "z") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract area of interest from data.
    
    Args:
        data: netCDF4 dataset
        aoi: Area of interest [lon_min, lon_max, lat_min, lat_max]
        datakey: Key for data variable in dataset
        
    Returns:
        Tuple of (data_slice, lat_grid, lon_grid)
    """
    vslice = slice(
        np.argmin(abs(data["lat"][:] - aoi[2])),
        np.argmin(abs(data["lat"][:] - aoi[3])),
    )
    z = data[datakey][vslice, :]
    lat = data["lat"][vslice]

    hslice = slice(
        np.argmin(abs(data["lon"][:] - aoi[0])),
        np.argmin(abs(data["lon"][:] - aoi[1])),
    )
    z = z[:, hslice]
    lon = data["lon"][hslice]

    lat, lon = np.meshgrid(lat, lon)
    return (z, lat.T, lon.T)


def compute_location_variables(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute location-based features from latitude and longitude.
    
    Args:
        lat: Latitude array
        lon: Longitude array
        
    Returns:
        Tuple of (rlat, lon_s, lon_c)
    """
    rlat = np.deg2rad(lat)
    lon_s = np.sin(np.deg2rad(lon))
    lon_c = np.cos(np.deg2rad(lon))
    return rlat, lon_s, lon_c


def load_distance_to_coast(dist_file_path: Union[str, Path], aoi: list, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Load and interpolate distance to coast data.
    
    Args:
        dist_file_path: Path to distance grid file
        aoi: Area of interest [lon_min, lon_max, lat_min, lat_max]
        lat: Latitude grid
        lon: Longitude grid
        
    Returns:
        Interpolated distance to coast grid
    """
    # Load distance grid from file
    grid_dist = np.load(dist_file_path, mmap_mode="r")

    # Define grid coordinates
    lats = np.arange(-90.045, 90.045 + 0.01 / 2, 0.01)
    lons = np.arange(-0.095, 360.095 + 0.01 / 2, 0.01)
    lons = lons[1:] + 0.01 / 2
    lats = lats[1:] + 0.01 / 2

    # Extract AOI from grid
    latlims = [
        int(np.argmin(abs(lats - aoi[2]))),
        int(np.argmin(abs(lats - aoi[3]))),
    ]
    lonlims = [
        int(np.argmin(abs(lons - aoi[0]))),
        int(np.argmin(abs(lons - aoi[1]))),
    ]

    grid_dist_aoi = np.copy(grid_dist[latlims[0] : latlims[1], lonlims[0] : lonlims[1]])

    # Create interpolation function
    lats_aoi = lats[latlims[0] : latlims[1]]
    lons_aoi = lons[lonlims[0] : lonlims[1]]

    grid_dist_intpf = scipy.interpolate.RegularGridInterpolator(
        (lats_aoi, lons_aoi),
        grid_dist_aoi,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    # Interpolate to data locations
    lon_column = np.copy(lon.reshape(-1))
    lon_column[lon_column < 0] += 360
    int_locations = np.array([lat.reshape(-1), lon_column]).T
    dist = grid_dist_intpf(int_locations).reshape(lon.shape)

    return dist


def filter_and_mask(
    data_dict: Dict, config: Dict
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Apply filtering and masking based on configuration.
    
    Args:
        data_dict: Dictionary with data arrays
        config: Filtering configuration
        
    Returns:
        Tuple of (filtered_data_dict, mask)
    """
    filtering = config.get("filtering", {})

    ship = data_dict["ship"]
    dist = data_dict["dist"]
    grav_on_topo = data_dict["grav_on_topo"]
    sio_vgg = data_dict["sio_vgg"]

    # Create mask
    mask = (
        (~np.isnan(ship))
        & (dist > filtering.get("min_dist_to_coast", 0.1))
        & (abs(grav_on_topo) < filtering.get("grav_on_topo_bounds", 300))
        & (abs(sio_vgg) < filtering.get("sio_bounds", 400))
    )

    # Apply mask to all arrays
    filtered_dict = {}
    for key, arr in data_dict.items():
        filtered_dict[key] = arr[mask]

    return filtered_dict, mask


def group_data(
    lon: np.ndarray,
    lat: np.ndarray,
    data_dict: Dict[str, np.ndarray],
    group_size: int,
    visualize: bool = False,
) -> np.ndarray:
    """Group data using KDTree with optional visualization.
    
    Args:
        lon: Longitude array (flattened)
        lat: Latitude array (flattened)
        data_dict: Dictionary with feature arrays (all flattened)
        group_size: Number of KDTree recursion levels
        visualize: Whether to generate visualization plots
        
    Returns:
        Grouped data array of shape (n_groups, group_size, n_features)
    """
    grouping_i = group_size

    points = np.asarray(np.c_[lon, lat])

    # Setup rotation matrix for KDTree
    deg = 45
    radians = np.radians(deg)
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, s], [-s, c]])

    # Generate KDTree with rotated points
    tree = cKDTree((j @ points.T).T, balanced_tree=True)

    # Navigate KDTree to create groups
    groups = []

    def recurse(node, i=0):
        g = node.greater
        l = node.lesser
        if i == grouping_i:
            groups.append([g.data_points, g.indices])
            groups.append([l.data_points, l.indices])
            return
        recurse(g, i=i + 1)
        recurse(l, i=i + 1)

    recurse(tree.tree)

    # Setup rotation matrix for visualization
    radians = np.radians(-deg)
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, s], [-s, c]])

    if visualize:
        plt.figure(figsize=(10, 8))
        list_len = []
        for group, indices in groups:
            points_rot_ = (j @ group.T).T
            list_len.append(len(group))
            plt.plot(points_rot_[:, 0], points_rot_[:, 1], "o", markersize=0.5)

        print(f"There are {len(groups)} groups.", end=" ")
        print(f"biggest - smallest = {max(list_len)} - {min(list_len)} = {max(list_len) - min(list_len)}")
        plt.title("Data Groups")
        plt.show()

    # Extract features and group
    n_per_group = min([len(group) for group, _ in groups])
    print(f"Using {n_per_group} samples in each group")

    grouped_data = []
    for points_, indices in groups:
        # Stack all features for this group
        group_features = np.c_[
            data_dict["rlat"][indices][:n_per_group],
            data_dict["lon_s"][indices][:n_per_group],
            data_dict["lon_c"][indices][:n_per_group],
            data_dict["grav_on_topo"][indices][:n_per_group],
            data_dict["topo_low"][indices][:n_per_group],
            data_dict["sio_vgg"][indices][:n_per_group],
            data_dict["dist"][indices][:n_per_group],
            data_dict["ship"][indices][:n_per_group],
        ]
        grouped_data.append(group_features)

    grouped_data = np.array(grouped_data)
    print(f"Data grouped, shape: {grouped_data.shape}")

    return grouped_data


def preprocess_data(config: Dict, visualize: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full preprocessing pipeline: load, slice, filter, group data.
    
    Args:
        config: Preprocessing configuration from YAML
        visualize: Whether to generate exploratory plots
        
    Returns:
        Tuple of (train_data, train_targets, prediction_data) as torch tensors
    """
    data_path = config.get("data_path")
    region = config.get("region", "global")
    aoi = config.get("aoi", [-180, 180, -90, 90])
    preprocessing_cfg = config.get("preprocessing", {})
    group_size = preprocessing_cfg.get("group_size", 10)

    print(f"Loading data from {data_path} for region {region}")

    # Load data
    datasets = load_netcdf_data(data_path)

    # Slice data to AOI
    grav_on_topo, lat, lon = slice_data(datasets["grav_on_topo"], aoi)
    topo_low, _, _ = slice_data(datasets["topo_low"], aoi)
    topo_ship, _, _ = slice_data(datasets["topo_ship"], aoi)
    topo_ship[topo_ship > -10] = np.nan
    grav_SWOT, _, _ = slice_data(datasets["grav_SWOT"], aoi)
    sio_vgg, _, _ = slice_data(datasets["sio_vgg"], aoi)

    # Compute location variables
    rlat, lon_s, lon_c = compute_location_variables(lat, lon)

    # Flatten arrays for processing
    grav_on_topo_flat = grav_on_topo.reshape(-1)
    ship_flat = topo_ship.reshape(-1)
    topo_low_flat = topo_low.reshape(-1)
    sio_vgg_flat = sio_vgg.reshape(-1)
    rlat_flat = rlat.reshape(-1)
    lon_s_flat = lon_s.reshape(-1)
    lon_c_flat = lon_c.reshape(-1)
    lon_flat = lon.reshape(-1)
    lat_flat = lat.reshape(-1)

    # TODO: Load distance to coast (requires external file path)
    # For now, create placeholder
    dist_flat = np.zeros_like(lat_flat)

    # Prepare data dictionary
    data_dict = {
        "grav_on_topo": grav_on_topo_flat,
        "ship": ship_flat,
        "topo_low": topo_low_flat,
        "sio_vgg": sio_vgg_flat,
        "rlat": rlat_flat,
        "lon_s": lon_s_flat,
        "lon_c": lon_c_flat,
        "lon": lon_flat,
        "lat": lat_flat,
        "dist": dist_flat,
    }

    # Filter data
    filtered_dict, mask = filter_and_mask(data_dict, config)

    print(f"Valid data points: {filtered_dict['ship'].size}")

    # Conditionally group data
    enable_grouping = preprocessing_cfg.get("group_data", False)
    
    if enable_grouping:
        print(f"[GROUPING] Applying spatial grouping (group_size={group_size})...")
        grouped_data = group_data(
            filtered_dict["lon"],
            filtered_dict["lat"],
            filtered_dict,
            group_size,
            visualize=visualize,
        )
        
        # Separate features and targets from grouped data
        # Features: all but last column (ship bathymetry)
        # Targets: last column (ship bathymetry)
        train_data = torch.from_numpy(grouped_data[:, :, :-1]).float()
        train_targets = torch.from_numpy(grouped_data[:, :, -1]).float()
    else:
        print("[GROUPING] Skipping spatial grouping - using flattened data")
        # Stack features without grouping
        train_data = torch.from_numpy(
            np.c_[
                filtered_dict["rlat"],
                filtered_dict["lon_s"],
                filtered_dict["lon_c"],
                filtered_dict["grav_on_topo"],
                filtered_dict["topo_low"],
                filtered_dict["sio_vgg"],
                filtered_dict["dist"],
            ]
        ).float()
        train_targets = torch.from_numpy(filtered_dict["ship"]).float()

    # For predictions, use all features (excluding ship data)
    # Reshape back to original grid for spatial predictions
    prediction_features = torch.from_numpy(
        np.c_[
            filtered_dict["rlat"],
            filtered_dict["lon_s"],
            filtered_dict["lon_c"],
            filtered_dict["grav_on_topo"],
            filtered_dict["topo_low"],
            filtered_dict["sio_vgg"],
            filtered_dict["dist"],
        ]
    ).float()

    print(f"Train data shape: {train_data.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Prediction data shape: {prediction_features.shape}")

    return train_data, train_targets, prediction_features


class BathymetryDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for bathymetry training."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor = None):
        """Initialize dataset.
        
        Args:
            data: Input features tensor (n_samples, n_features)
            targets: Target values tensor (n_samples,) or None for inference
        """
        self.data = data
        self.targets = targets
        self.input_dim = data.shape[-1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.targets is None:
            return self.data[idx]
        return self.data[idx], self.targets[idx]


def get_data_loaders(
    data: torch.Tensor,
    targets: torch.Tensor,
    config: Dict,
    device: str,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation data loaders.
    
    Args:
        data: Training data tensor
        targets: Training targets tensor
        config: Data configuration
        device: Device ("cuda" or "cpu")
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_cfg = config.get("data", {})
    train_batch_size = data_cfg.get("train_minibatch_size", 2076)
    val_block_size = data_cfg.get("validation_block_size", 10)

    # Split into train and validation
    val_size = train_batch_size * val_block_size
    train_data = data[:-val_size]
    train_targets = targets[:-val_size]
    val_data = data[-val_size:]
    val_targets = targets[-val_size:]

    # Move to device
    if device == "cuda":
        train_data = train_data.cuda()
        train_targets = train_targets.cuda()
        val_data = val_data.cuda()
        val_targets = val_targets.cuda()

    # Create datasets
    train_dataset = BathymetryDataset(train_data, train_targets)
    val_dataset = BathymetryDataset(val_data, val_targets)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


def visualize_data_distribution(
    grouped_data: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
    n_skip: int = 100,
):
    """Create exploratory plots of data distribution.
    
    Args:
        grouped_data: Grouped data array from preprocessing
        save_path: Path to save plots (optional)
        show: Whether to display plots interactively
        n_skip: Sampling factor for scatter plots
    """
    # Extract data for visualization
    # grouped_data shape: (n_groups, group_size, n_features)
    # Reshape to (total_samples, n_features)
    total_samples = grouped_data.shape[0] * grouped_data.shape[1]
    data_flat = grouped_data.reshape(total_samples, -1)

    feature_names = ["rlat", "lon_s", "lon_c", "grav_on_topo", "topo_low", "sio_vgg", "dist", "ship"]

    for i, name in enumerate(feature_names):
        if i < data_flat.shape[1]:
            plt.figure()
            plt.hist(data_flat[::n_skip, i], bins=100)
            plt.title(f"Distribution of {name}")
            plt.xlabel(name)
            plt.ylabel("Count")

            if save_path:
                save_file = Path(save_path) / f"hist_{name}.png"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_file)

            if show:
                plt.show()
            else:
                plt.close()


def exploratory_run(config: Dict):
    """Full preprocessing pipeline with visualization.
    
    Args:
        config: Preprocessing configuration from YAML
    """
    print("Running exploratory preprocessing with visualization...")

    # Update config to enable visualization
    config["visualization"]["enabled"] = True

    train_data, train_targets, pred_data = preprocess_data(config, visualize=True)

    # Generate additional visualizations
    if config.get("visualization", {}).get("save_plots", True):
        output_dir = resolve_path(config.get("visualization", {}).get("plots_dir", "reports/figures/"))
        print(f"Saving plots to {output_dir}")

        # Combine for visualization
        grouped_with_targets = np.concatenate([train_data, train_targets.unsqueeze(-1)], dim=-1)

        visualize_data_distribution(
            grouped_with_targets.numpy(),
            save_path=output_dir,
            show=config.get("visualization", {}).get("show_plots", False),
        )

    print("Exploratory run complete!")
