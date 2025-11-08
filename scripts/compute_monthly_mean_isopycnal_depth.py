"""Compute the monthly mean isopycnal depth from saved isopycnal depths.

Intended to be run after `compute_isopycnal_depth.py` to compute monthly means.
"""

from pathlib import Path

import xarray as xr


def compute_monthly_mean_isopycnal_depth(isopycnal_depth_path: str | Path) -> None:
    """Compute and save the monthly mean isopycnal depth from saved isopycnal depth slices.

    Args:
        isopycnal_depth_path (str | Path): Path to the zarr directory containing isopycnal depth slices.

    """
    if isinstance(isopycnal_depth_path, str):
        isopycnal_depth_path = Path(isopycnal_depth_path)

    # Open the zarr dataset
    ds = xr.open_zarr(isopycnal_depth_path)

    # Resample to monthly means
    monthly_mean_ds = ds.groupby("time.month").mean(dim="time")

    # Save the monthly mean dataset
    save_path = isopycnal_depth_path.parent / f"monthly_mean_{isopycnal_depth_path.name}"
    monthly_mean_ds.to_zarr(save_path)
    print(f"Monthly mean isopycnal depth saved to {save_path}.")


if __name__ == "__main__":
    compute_monthly_mean_isopycnal_depth("D:/avg/isopycnal_depth_25.8.zarr")
