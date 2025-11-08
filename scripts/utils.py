"""Utility functions for handling Oceanic Pathways model data."""

from pathlib import Path

import xarray as xr


def open_model_fields(parent_path: str) -> xr.Dataset:
    """Open Oceanic Pathways model fields from a parent directory.

    Args:
        parent_path (str): Path to the parent directory containing model output files.

    Returns:
        xr.Dataset: An xarray Dataset containing the model fields.

    """
    # Find all files with full depth data (-complete.nc)
    model_files = list(Path(parent_path).rglob("*-complete.nc"))

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


def open_grid(parent_path: str) -> xr.Dataset:
    """Open the grid file from a parent directory.

    Args:
        parent_path (str): Path to the parent directory containing the grid file.

    Returns:
        xr.Dataset: An xarray Dataset containing the grid information.

    """
    grid_file = Path(parent_path) / "croco_grd.nc.1b"
    ds_grid = xr.open_dataset(grid_file)
    return ds_grid


def open_grid_with_zdepths(parent_path: str) -> xr.Dataset:
    """Open the grid file with z-depths from a parent directory.

    Args:
        parent_path (str): Path to the parent directory containing the grid file with z-depths.

    Returns:
        xr.Dataset: An xarray Dataset containing the grid information with z-depths.

    """
    grid_file = Path(parent_path) / "croco_grd_with_z.nc"
    ds_grid = xr.open_dataset(grid_file)
    return ds_grid
