"""Utility functions for handling Oceanic Pathways model data."""

from pathlib import Path

import xarray as xr


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

    isopycnal_depth_path = parent_path / f"isopycnal_depth_{target_sigma_0}.zarr"
    return isopycnal_depth_path
