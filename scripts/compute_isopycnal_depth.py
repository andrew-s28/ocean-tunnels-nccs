"""Computes the depth of a specified isopycnal surface (e.g., sigma_0 = 25.8 kg/m^3) from CROCO model output.

Intended to be used after running `compute_z_levels.py` to ensure z-depths and pressures are available.
"""

import shutil
from pathlib import Path

import gsw
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client, LocalCluster
from scipy.optimize import brentq as find_root
from tqdm import tqdm

from utils import open_grid_with_zdepths, open_model_fields


def interpolate_to_density_level(z: np.ndarray, sigma_0: np.ndarray, target_sigma_0: float = 25.8) -> float:
    """Interpolate to find the depth at which the given sigma0 matches the target sigma0.

    Intended to be used with the xarray.apply_ufunc defined within this script.

    Args:
        z (np.ndarray): Array of depths.
        sigma_0 (np.ndarray): Array of sigma_0 values corresponding to the depths.
        target_sigma_0 (float): The target sigma_0 value to interpolate to.

    Returns:
        float: The depth at which sigma_0 matches target_sigma_0, or np.nan if not found.

    """
    try:
        root = find_root(
            lambda depth: np.interp(depth, z, sigma_0, left=np.nan, right=np.nan) - target_sigma_0,
            -6000,
            0,
        )
    except ValueError:
        root = np.nan
    # Satisfy type checker
    if type(root) is float:
        return root
    return np.nan


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
    ds["sigma0"] = gsw.density.sigma0(
        ds["salt_abs"],
        ds["temp_con"],
    )
    return ds


def _setup_cluster() -> Client:
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="8GiB")
    client = Client(cluster)
    return client


if __name__ == "__main__":
    """Main processing function when script is run directly.

    Computes the depth of a specified isopycnal surface (e.g., sigma_0 = 25.8 kg/m^3) from CROCO model output.

    Can adjust the chunk sizes, paths, and target density level as needed.

    Takes about four hours to run on a home desktop and saves a bit over 2GB of data
    when used on the Oceanic Pathways model output.
    """

    parent_path = "D:/avg"
    slices_dir = Path(parent_path) / "isopycnal_depth_25.8"
    slices_dir.mkdir(exist_ok=True)

    with _setup_cluster() as client:
        print(f"Dask cluster setup. Dashboard link: {client.dashboard_link}")

        print("Opening model fields and grid...", end="", flush=True)
        # Open model fields and grid with z-depths computed
        ds = open_model_fields(parent_path)
        grid = open_grid_with_zdepths(parent_path)
        print("done.")

        # Select a target density level (e.g., sigma0 = 25.8 kg/m^3)
        target_sigma0 = 25.8

        # Split the dataset into manageable chunks of time
        time_slice_size = 100
        number_of_slices = int(np.ceil(len(ds["time"]) / time_slice_size))

        # Process each chunk of times separately
        for i in tqdm(range(number_of_slices)):
            start_idx = i * time_slice_size
            end_idx = min((i + 1) * time_slice_size, len(ds["time"]))
            ds_slice = ds.isel(time=slice(start_idx, end_idx))

            ds_slice = compute_sigma_0(ds_slice, grid)

            # Apply the interpolation function across the dataset slice
            isopycnal_depth_slice: xr.DataArray = xr.apply_ufunc(
                interpolate_to_density_level,
                grid["z_rho"],
                ds_slice["sigma0"],
                input_core_dims=[["s_rho"], ["s_rho"]],
                vectorize=True,
                dask="parallelized",
            )

            # Save the computed isopycnal depths for this slice to a separate NetCDF file, ~2GB total
            print(f"Computing and saving isopycnal depths for slice {i + 1} to NetCDF...", end="", flush=True)
            save_path = slices_dir / f"isopycnal_depth_25.8_slice_{i + 1}.zarr"
            if save_path.exists():
                print(f"File {save_path} already exists. Skipping computation.")
                continue
            with ProgressBar():
                isopycnal_depth_slice.to_zarr(save_path)
            print("done.")

        print("Concatenating all slices into a single dataset...", end="", flush=True)
        # Open all saved slices and concatenate them into a single dataset
        slice_files = list(slices_dir.glob("isopycnal_depth_25.8_slice_*.zarr"))
        slice_files.sort()
        isopycnal_depth = xr.concat(
            [xr.open_zarr(f) for f in slice_files],
            dim="time",
            compat="no_conflicts",
            coords="minimal",
        )["__xarray_dataarray_variable__"]
        isopycnal_depth.to_zarr(Path(parent_path) / "isopycnal_depth_25.8.zarr")
        print("done.")

        print("Cleaning up slice files...", end="", flush=True)
        [shutil.rmtree(f) for f in slice_files]
        print("done.")

        print("All processing complete.")
        client.close()
