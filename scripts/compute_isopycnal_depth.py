"""Computes the depth of a specified isopycnal surface (e.g., sigma_0 = 25.8 kg/m^3) from CROCO model output.

Intended to be used after running `compute_z_levels.py` to ensure z-depths and pressures are available.
"""

from pathlib import Path

import gsw
import numpy as np
import xarray as xr
from scipy.optimize import brentq as find_root
from tqdm import tqdm

from utils import (
    compute_sigma_0,
    concatenate_slices,
    get_isopycnal_depth_path,
    open_grid_with_zdepths,
    open_model_fields,
    save_slice,
    setup_cluster,
    slice_dataset,
)


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
            np.min(z),
            np.max(z),
        )
    except ValueError:
        root = np.nan
    # Satisfy type checker, find_root should always return a float if no exception is raised
    if type(root) is float:
        return root
    return np.nan


def process_chunk(ds_chunk: xr.Dataset, grid: xr.Dataset, target_sigma_0: float) -> xr.DataArray:
    """Process a chunk of the dataset to compute isopycnal depths.

    Args:
        ds_chunk (xr.Dataset): Chunk of the model dataset.
        grid (xr.Dataset): CROCO grid with z-depths.
        target_sigma_0 (float): The target sigma_0 value to compute depths for.

    Returns:
        xr.DataArray: DataArray containing the computed isopycnal depths for the chunk.

    """
    ds_chunk = compute_sigma_0(ds_chunk, grid)

    isopycnal_depth_chunk: xr.DataArray = xr.apply_ufunc(
        interpolate_to_density_level,
        grid["z_rho"],
        ds_chunk["sigma0"],
        input_core_dims=[["s_rho"], ["s_rho"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[grid["z_rho"].dtype],
        kwargs={"target_sigma_0": target_sigma_0},
    )
    isopycnal_depth_chunk = isopycnal_depth_chunk.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1})
    isopycnal_depth_chunk = isopycnal_depth_chunk.rename("depth")

    return isopycnal_depth_chunk


def compute_isopycnal_depth(parent_path: str | Path, target_sigma_0: float, time_slice_size: int) -> None:
    """Compute the depth of a specified isopycnal surface (e.g., sigma_0 = 25.8 kg/m^3) from CROCO model output.

    Takes about 20 hours to run on a home desktop and saves ~20 GB of data
    when used on the Oceanic Pathways model output.

    Args:
        parent_path (str | Path): Path to the parent directory containing model output files.
        target_sigma_0 (float): The target sigma_0 value to compute depths for.
        time_slice_size (int): Number of time steps to process in each slice.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    slices_dir = get_isopycnal_depth_path(parent_path, target_sigma_0).with_name("isopycnal_depth_slices")
    slices_dir.mkdir(exist_ok=True)

    with setup_cluster(n_workers=4, threads_per_worker=2, memory_limit="8GiB") as client:
        print(f"Dask cluster setup. Dashboard link: {client.dashboard_link}")

        print("Opening model fields and grid...", end="", flush=True)
        # Open model fields and grid with z-depths computed
        ds = open_model_fields(parent_path)
        grid = open_grid_with_zdepths(parent_path)
        print("done.")

        # Find the number of time slices needed
        dataset_slices = slice_dataset(ds, time_slice_size)

        print("Computing and saving isopycnal depth slices...", end="", flush=True)

        # Process each chunk of times separately
        for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing isopycnal depth slices")):
            isopycnal_depth_slice = process_chunk(ds_slice, grid, target_sigma_0)

            # Save the computed isopycnal depths for this slice to a separate NetCDF file, ~1GB total per slice
            save_path = slices_dir / f"isopycnal_depth_slice_{i + 1}.zarr"
            if save_path.exists():
                tqdm.write(f"File {save_path} already exists. Skipping computation.")
                continue
            save_slice(isopycnal_depth_slice, save_path)

        print("done.")

        save_path = get_isopycnal_depth_path(parent_path, target_sigma_0)
        concatenate_slices(save_path, slices_dir)

        print("All processing complete.")
        client.close()


if __name__ == "__main__":
    compute_isopycnal_depth(parent_path="D:/avg", target_sigma_0=25.8, time_slice_size=100)
