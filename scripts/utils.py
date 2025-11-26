"""Utility functions for handling Oceanic Pathways model data."""

import shutil
import warnings
import webbrowser
from pathlib import Path

import gsw
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from zarr.errors import ZarrUserWarning


def compute_sigma_0(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """Compute sigma_0 from practical salinity and potential temperature.

    Run `compute_z_levels.py` first to calculate rho-pressure levels if needed.

    Args:
        ds (xr.Dataset): CROCO model with salinity, temperature, and pressure data.
        grid (xr.Dataset): CROCO grid with longitude and latitude data.

    Returns:
        xr.Dataset: Dataset with added sigma_0 variable.

    """
    ds["salt_abs"] = gsw.conversions.SA_from_SP(
        ds["salt"],
        grid["p_rho"],
        grid["lon_rho"],
        grid["lat_rho"],
    )
    ds["temp_con"] = gsw.conversions.CT_from_pt(
        ds["salt_abs"],
        ds["temp"],
    )
    ds["sigma_0"] = gsw.density.sigma0(
        ds["salt_abs"],
        ds["temp_con"],
    )
    return ds


def setup_client(n_workers: int, threads_per_worker: int, memory_limit: str) -> Client:
    """Set up a Dask distributed cluster for parallel processing.

    Args:
        n_workers (int): Number of workers in the cluster.
        threads_per_worker (int): Number of threads per worker.
        memory_limit (str): Memory limit per worker (e.g., '4GB').

    Returns:
        Client: A Dask distributed client connected to the cluster.

    """
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
    client = Client(cluster)
    print(f"Dask cluster setup. Opening dashboard at: {client.dashboard_link}")
    webbrowser.open_new(client.dashboard_link) if client.dashboard_link else None

    return client


def save_slice(
    slice_da: xr.DataArray,
    save_path: Path,
) -> None:
    """Save a computed isopycnal depth slice to a zarr file.

    Args:
        slice_da (xr.DataArray): DataArray containing the computed depths for the slice.
        save_path (Path): Path to save the zarr file.

    """
    with warnings.catch_warnings():
        msg = "Consolidated metadata is currently not part in the Zarr format 3 specification"
        warnings.filterwarnings("ignore", category=ZarrUserWarning, message=msg)
        msg = "Sending large graph of size"
        warnings.filterwarnings("ignore", category=UserWarning, message=msg)
        slice_da.to_zarr(save_path)


def slice_dataset(
    ds: xr.Dataset,
    slice_size: int,
) -> list[xr.Dataset]:
    """Slice a dataset into smaller datasets along the time dimension.

    Args:
        ds (xr.Dataset): The input dataset to be sliced.
        slice_size (int): The number of time steps in each slice.

    Returns:
        list[xr.Dataset]: A list of sliced datasets.

    """
    # Find the number of time slices needed
    number_of_slices = int(np.ceil(len(ds["time"]) / slice_size))
    slices = [
        ds.isel(time=slice(i * slice_size, min((i + 1) * slice_size, len(ds.time)))) for i in range(number_of_slices)
    ]
    return slices


def concatenate_slices(save_path: Path | str, slices_dir: Path | str) -> None:
    """Concatenate all saved slices into a single dataset and remove the slice files.

    Args:
        save_path (Path | str): Path to save the concatenated dataset.
        slices_dir (Path | str): Directory containing the saved isopycnal depth slices.

    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if isinstance(slices_dir, str):
        slices_dir = Path(slices_dir)

    print("Concatenating all slices into a single dataset...", end="", flush=True)
    slice_files = list(slices_dir.glob("*slice_*.zarr"))
    slice_files.sort()
    ds = xr.concat(
        [xr.open_zarr(f) for f in slice_files],
        dim="time",
        compat="no_conflicts",
        coords="minimal",
    )
    with warnings.catch_warnings():
        msg = "Consolidated metadata is currently not part in the Zarr format 3 specification"
        warnings.filterwarnings("ignore", category=ZarrUserWarning, message=msg)
        msg = "Sending large graph of size"
        warnings.filterwarnings("ignore", category=UserWarning, message=msg)
        ds.to_zarr(save_path)
    print("done.")

    print("Cleaning up slice files...", end="", flush=True)
    [shutil.rmtree(f) for f in slice_files]
    slices_dir.rmdir()
    print("done.")


def open_model_fields(parent_path: str | Path) -> xr.Dataset:
    """Open Oceanic Pathways model fields from a parent directory.

    Args:
        parent_path (str | Path): Path to the parent directory containing model output files.

    Returns:
        xr.Dataset: An xarray Dataset containing the model fields.

    Raises:
        FileNotFoundError: If no model files are found in the specified directory.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    # Find all files with full depth data (-complete.nc)
    model_files = list(parent_path.rglob("*-complete.nc"))

    if not model_files:
        msg = f"No model files found in {parent_path}."
        raise FileNotFoundError(msg)

    # Sort files to ensure correct time order
    model_files.sort()

    # Filter out files from 1979
    model_files = [f for f in model_files if not f.name.startswith("1979")]

    ds = xr.open_mfdataset(
        model_files,
        combine="by_coords",
        data_vars="minimal",
        compat="equals",
        chunks={"time": 1, "s_rho": -1, "eta_rho": -1, "xi_rho": -1},
    )

    return ds


def open_grid(parent_path: str | Path) -> xr.Dataset:
    """Open the grid file from a parent directory.

    Args:
        parent_path (str | Path): Path to the parent directory containing the grid file.

    Returns:
        xr.Dataset: An xarray Dataset containing the grid information.

    Raises:
        FileNotFoundError: If the grid file is not found in the specified directory.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    grid_file = parent_path / "croco_grd.nc.1b"

    if not grid_file.exists():
        msg = f"Grid file {grid_file.name} not found in directory {parent_path}."
        raise FileNotFoundError(msg)

    ds_grid = xr.open_dataset(grid_file)
    return ds_grid


def open_grid_with_zdepths(parent_path: str | Path) -> xr.Dataset:
    """Open the grid file with z-depths from a parent directory.

    Args:
        parent_path (str | Path): Path to the parent directory containing the grid file with z-depths.

    Returns:
        xr.Dataset: An xarray Dataset containing the grid information with z-depths.

    Raises:
        FileNotFoundError: If the grid file with z-depths is not found in the specified directory.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    grid_file = parent_path / "croco_grd_with_z.nc"

    if not grid_file.exists():
        msg = f"Grid file with z-depths {grid_file.name} not found in directory {parent_path}."
        raise FileNotFoundError(msg)

    ds_grid = xr.open_dataset(grid_file)
    return ds_grid


def get_mixed_layer_depth_path(parent_path: str | Path, delta_sigma: float) -> Path:
    """Get the path to the mixed layer depth zarr directory.

    Args:
        parent_path (str | Path): Path to the parent directory containing the mixed layer depth data.
        delta_sigma (float): The sigma_0 threshold for mixed layer depth calculation.

    Returns:
        Path: The path to the mixed layer depth zarr directory.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    mld_path = parent_path / f"mixed_layer_depth_delta_sigma_{delta_sigma}.zarr"
    return mld_path


def get_isopycnal_depth_path(parent_path: str | Path, target_sigma_0: float) -> Path:
    """Get the path to the isopycnal depth zarr directory.

    Args:
        parent_path (str | Path): Path to the parent directory containing the isopycnal depth data.
        target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.

    Returns:
        Path: The path to the isopycnal depth zarr directory.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    isopycnal_depth_path = parent_path / f"isopycnal_depth_sigma_{target_sigma_0}.zarr"
    return isopycnal_depth_path


def get_monthly_mean_isopycnal_depth_path(parent_path: str | Path, target_sigma_0: float) -> Path:
    """Get the path to the monthly mean isopycnal depth zarr directory.

    Args:
        parent_path (str | Path): Path to the parent directory containing the monthly mean isopycnal depth data.
        target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.

    Returns:
        Path: The path to the monthly mean isopycnal depth zarr directory.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    isopycnal_depth_path = get_isopycnal_depth_path(parent_path, target_sigma_0)
    monthly_mean_path = parent_path / f"monthly_mean_{isopycnal_depth_path.name}"
    return monthly_mean_path


def open_isopycnal_depth(parent_path: str | Path, target_sigma_0: float) -> xr.Dataset:
    """Open the isopycnal depth zarr dataset.

    Args:
        parent_path (str | Path): Path to the parent directory containing the isopycnal depth data.
        target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.

    Returns:
        xr.Dataset: An xarray Dataset containing the isopycnal depth data.

    """
    isopycnal_depth_path = get_isopycnal_depth_path(parent_path, target_sigma_0)
    ds_isopycnal_depth = xr.open_zarr(isopycnal_depth_path)
    return ds_isopycnal_depth


def open_monthly_mean_isopycnal_depth(parent_path: str | Path, target_sigma_0: float) -> xr.Dataset:
    """Open the monthly mean isopycnal depth zarr dataset.

    Args:
        parent_path (str | Path): Path to the parent directory containing the monthly mean isopycnal depth data.
        target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.

    Returns:
        xr.Dataset: An xarray Dataset containing the monthly mean isopycnal depth data.

    """
    monthly_mean_path = get_monthly_mean_isopycnal_depth_path(parent_path, target_sigma_0)

    ds_monthly_mean = xr.open_zarr(monthly_mean_path)
    return ds_monthly_mean
