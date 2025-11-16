"""Analyze profiler data from the CE09 Washington Offshore McLane Moored Profiler.

Data retrieved from Risien et al., 2025 https://zenodo.org/records/15627742

Contains code to compute mixed layer depth and isopycnal layer depths from profiler data.
"""

from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy.optimize import brentq as find_root

DATA_DIRECTORY = Path("../data")
CE09_OSPM_FILE = DATA_DIRECTORY / "ce09ospm_gridded_profiles.nc"


def add_attributes_to_dataarray(
    da: xr.DataArray,
    units: str,
    long_name: str,
    description: str,
    **kwargs: Any,  # noqa: ANN401; xarray is fine with Any for attribute values
) -> xr.DataArray:
    """Add standard attributes to a DataArray.

    Args:
        da (xr.DataArray): DataArray to add attributes to.
        units (str): Units of the DataArray.
        long_name (str): Long name of the DataArray.
        description (str): Description of the DataArray.
        **kwargs: Additional attributes to add in key-value pairs.

    Returns:
        xr.DataArray: DataArray with added attributes.

    """
    da.attrs["units"] = units
    da.attrs["long_name"] = long_name
    da.attrs["description"] = description
    da.attrs["time_created"] = datetime.now(tz=UTC).isoformat()
    for key, value in kwargs.items():
        da.attrs[key] = value
    return da


def compute_mld_for_profile(
    depth: np.ndarray,
    sigma: np.ndarray,
    target_sigma: float,
) -> float:
    """Compute mixed layer depth for a single profile.

    Args:
        depth (np.ndarray): Array of depths at which to calculate the mixed layer depth.
        sigma (np.ndarray): Array of sigma values.
        target_sigma (float): Target sigma value for mixed layer depth calculation.

    Returns:
        float: Mixed layer depth.

    """

    def _mld_func(z: np.ndarray, depth: np.ndarray, sigma: np.ndarray, target_sigma: float) -> float:
        mld = np.interp(z, depth, sigma, left=np.nan, right=np.nan) - target_sigma
        return float(mld)

    try:
        mld = find_root(
            partial(
                _mld_func,
                depth=depth,
                sigma=sigma,
                target_sigma=target_sigma,
            ),
            np.min(depth),
            np.max(depth),
        )
    except ValueError:
        mld = np.nan
    # Satisfy type checker, find_root should always return a float if no exception is raised
    if type(mld) is float:
        return mld
    return np.nan


def calculate_mixed_layer_depth(
    ds: xr.Dataset,
    threshold: float,
    density_name: str = "potential_density",
    depth_name: str = "depth",
) -> xr.DataArray:
    """Calculate mixed layer depth from profiler dataset using the threshold method.

    Note that, since the CE09 profiler has a minimum depth of 30 m, this method cannot
    resolve mixed layer depths shallower than 30 m.

    Args:
        ds (xr.Dataset): Dataset containing potential density and depth.
        threshold (float): Threshold for mixed layer depth calculation.
        density_name (str): Name of the potential density variable in the dataset. Default is "potential_density".
        depth_name (str): Name of the depth variable in the dataset. Default is "depth".

    Returns:
        xr.DataArray: DataArray containing the mixed layer depth.

    """
    mld = np.empty(ds["time"].size) * np.nan
    for i, (sigma, target_sigma) in enumerate(
        zip(
            ds[density_name].isel(depth=slice(None)).to_numpy().T,
            (ds[density_name].isel(depth=0).to_numpy() + threshold),
            strict=True,
        ),
    ):
        depth_filtered = ds[depth_name].to_numpy()[~np.isnan(sigma)]
        sigma_filtered = sigma[~np.isnan(sigma)]
        mld[i] = compute_mld_for_profile(depth=depth_filtered, sigma=sigma_filtered, target_sigma=target_sigma)

    mld_da = xr.DataArray(mld, dims="time", name="mld")
    mld_da = add_attributes_to_dataarray(
        mld_da,
        units="m",
        long_name="mixed_layer_depth",
        description="Mixed layer depth calculated using the threshold method.",
        kwargs={
            "threshold": threshold,
            "source": CE09_OSPM_FILE.name,
            "author": "Andrew Scherer",
            "created_by": Path(__file__).name,
        },
    )
    return mld_da


def calculate_mixed_layer_depth_for_ce09(threshold: float) -> xr.DataArray:
    """Calculate mixed layer depth for CE09 Washington Offshore McLane Moored Profiler data.

    Saves the mixed layer depth to a new NetCDF file.

    Args:
        threshold (float): Threshold for mixed layer depth calculation.

    Returns:
        xr.DataArray: DataArray containing the mixed layer depth.

    """
    ce09 = xr.open_dataset(CE09_OSPM_FILE)
    ce09 = ce09.squeeze().load()
    ce09 = ce09.swap_dims({"pressure": "depth"})
    # backfill surface data for identifying surface sigma_0, but limit to 5 m depth (10 data points at 0.5 m intervals)
    ce09_bfill = ce09.bfill("depth", limit=10)
    mld = calculate_mixed_layer_depth(ce09_bfill, threshold)
    return mld


def calculate_isopycnal_layer_depth(ds: xr.Dataset, target_sigma: float) -> xr.DataArray:
    """Calculate isopycnal layer depth from profiler dataset.

    Args:
        ds (xr.Dataset): Dataset containing potential density and depth.
        target_sigma (float): Target sigma value for isopycnal layer depth calculation.

    Returns:
        xr.DataArray: DataArray containing the isopycnal layer depth.

    """
    isopycnal_depth = np.empty(ds["time"].size) * np.nan
    for i, sigma in enumerate(ds["potential_density"].isel(depth=slice(None)).to_numpy().T):
        depth_filtered = ds["depth"].to_numpy()[~np.isnan(sigma)]
        sigma_filtered = sigma[~np.isnan(sigma)]
        isopycnal_depth[i] = compute_mld_for_profile(
            depth=depth_filtered,
            sigma=sigma_filtered,
            target_sigma=target_sigma,
        )

    isopycnal_depth_da = xr.DataArray(isopycnal_depth, dims="time", name="isopycnal_depth")
    isopycnal_depth_da = add_attributes_to_dataarray(
        isopycnal_depth_da,
        units="m",
        long_name="isopycnal_layer_depth",
        description=f"Depth of the {target_sigma} kg/m^3 isopycnal layer.",
        kwargs={
            "source": CE09_OSPM_FILE.name,
            "author": "Andrew Scherer",
            "created_by": Path(__file__).name,
        },
    )
    return isopycnal_depth_da
