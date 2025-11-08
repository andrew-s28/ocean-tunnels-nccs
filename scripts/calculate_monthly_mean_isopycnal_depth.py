"""Calculate the monthly mean isopycnal depth from saved isopycnal depths.

Intended to be run after `compute_isopycnal_depth.py` to compute monthly means.
"""

import xarray as xr


def calculate_monthly_mean_isopycnal_depth(zarr_path: str) -> xr.Dataset:
    """Calculate the monthly mean isopycnal depth from saved isopycnal depth slices.

    Args:
        zarr_path (str): Path to the zarr directory containing isopycnal depth slices.

    Returns:
        xr.Dataset: Dataset containing the monthly mean isopycnal depth.

    """
    # Open the zarr dataset
    ds = xr.open_zarr(zarr_path)

    # Resample to monthly means
    monthly_mean_ds = ds.groupby("time.month").mean(dim="time")

    return monthly_mean_ds


def save_monthly_mean_isopycnal_depth() -> None:
    """Calculate and save the monthly mean isopycnal depth to a zarr directory."""
    zarr_path = "D:/avg/isopycnal_depth_25.8.zarr"
    monthly_mean_ds = calculate_monthly_mean_isopycnal_depth(zarr_path)

    # Save the monthly mean dataset to another zarr directory
    save_path = "D:/avg/monthly_mean_isopycnal_depth_25.8.zarr"
    monthly_mean_ds.to_zarr(save_path)
    print(f"Monthly mean isopycnal depth saved to {save_path}.")


if __name__ == "__main__":
    save_monthly_mean_isopycnal_depth()
