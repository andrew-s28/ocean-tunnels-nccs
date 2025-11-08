"""Compute the monthly mean isopycnal depth from saved isopycnal depths.

Intended to be run after `compute_isopycnal_depth.py` to compute monthly means.
"""

from pathlib import Path

import xarray as xr

from utils import get_isopycnal_depth_path, get_monthly_mean_isopycnal_depth_path


def compute_monthly_mean_isopycnal_depth(parent_path: str | Path, target_sigma_0: float) -> None:
    """Compute and save the monthly mean isopycnal depth from saved isopycnal depth slices.

    Args:
        parent_path (str | Path): Path to the parent directory containing isopycnal depth slices.
        target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    isopycnal_depth_path = get_isopycnal_depth_path(parent_path, target_sigma_0)

    # Open the zarr dataset
    ds = xr.open_zarr(isopycnal_depth_path)

    # Resample to monthly means
    monthly_mean_ds = ds.groupby("time.month").mean(dim="time")

    # Save the monthly mean dataset
    monthly_mean_path = get_monthly_mean_isopycnal_depth_path(parent_path, target_sigma_0)
    monthly_mean_ds.to_zarr(monthly_mean_path)
    print(f"Monthly mean isopycnal depth saved to {monthly_mean_path}.")


if __name__ == "__main__":
    compute_monthly_mean_isopycnal_depth("D:/avg", 25.8)
