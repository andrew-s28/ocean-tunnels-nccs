"""Computes the mixed layer depth based on the threshold method from CROCO model output."""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.optimize import brentq as find_root
from tqdm import tqdm

from utils import (
    compute_sigma_0,
    concatenate_slices,
    get_mixed_layer_depth_path,
    open_grid_with_zdepths,
    open_model_fields,
    save_slice,
    setup_cluster,
    slice_dataset,
)


def interpolate_to_mld_level(z: np.ndarray, sigma_0: np.ndarray, surface_sigma_0: np.ndarray) -> float:
    """Interpolate to find the depth at which the given sigma0 matches the target sigma0.

    Intended to be used with the xarray.apply_ufunc defined within this script.

    Args:
        z (np.ndarray): Array of depths.
        sigma_0 (np.ndarray): Array of sigma_0 values corresponding to the depths.
        surface_sigma_0 (np.ndarray): The target sigma_0 value to interpolate to.

    Returns:
        float: The depth at which sigma_0 matches surface_sigma_0, or np.nan if not found.

    """
    try:
        root = find_root(
            lambda depth: np.interp(depth, z, sigma_0, left=np.nan, right=np.nan) - surface_sigma_0,
            np.min(z),
            np.max(z),
        )
    except ValueError:
        root = np.nan
    # Satisfy type checker, find_root should always return a float if no exception is raised
    if type(root) is float:
        return root
    return np.nan


def process_mld_chunk(ds_chunk: xr.Dataset, grid: xr.Dataset, delta_sigma_0: float) -> xr.DataArray:
    """Process a chunk of the dataset to compute isopycnal depths.

    Args:
        ds_chunk (xr.Dataset): Chunk of the model dataset.
        grid (xr.Dataset): CROCO grid with z-depths.
        delta_sigma_0 (float): The sigma_0 threshold for mixed layer depth calculation.

    Returns:
        xr.DataArray: DataArray containing the computed isopycnal depths for the chunk.

    """
    ds_chunk = compute_sigma_0(ds_chunk, grid)
    ds_chunk["threshold_sigma_0"] = ds_chunk["sigma0"].isel(s_rho=-1) + delta_sigma_0  # grid is from bottom to top

    mixed_layer_depth_chunk: xr.DataArray = xr.apply_ufunc(
        interpolate_to_mld_level,
        grid["z_rho"],
        ds_chunk["sigma0"],
        ds_chunk["threshold_sigma_0"],
        input_core_dims=[["s_rho"], ["s_rho"], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[grid["z_rho"].dtype],
    )
    mixed_layer_depth_chunk = mixed_layer_depth_chunk.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1})
    mixed_layer_depth_chunk = mixed_layer_depth_chunk.rename("depth")

    return mixed_layer_depth_chunk


def compute_mixed_layer_depth(parent_path: str | Path, delta_sigma: float = 0.2, time_slice_size: int = 100) -> None:
    """Compute and save the mixed layer depth from CROCO model output using the threshold method.

    Args:
        parent_path (str | Path): Path to the parent directory containing CROCO model output.
        delta_sigma (float): The sigma_0 threshold for mixed layer depth calculation.
        time_slice_size (int): Number of time steps to process in each slice.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    slices_dir = get_mixed_layer_depth_path(parent_path, delta_sigma).with_name("mixed_layer_depth_slices")
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

        print("Computing and saving mixed layer depth slices...", end="", flush=True)

        # Process each chunk of times separately
        for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing mixed layer depth slices")):
            mixed_layer_depth_slice = process_mld_chunk(ds_slice, grid, delta_sigma)

            # Save the computed mixed layer depths for this slice to a separate zarr file
            save_path = slices_dir / f"mixed_layer_depth_slice_{i + 1}.zarr"
            if save_path.exists():
                tqdm.write(f"File {save_path} already exists. Skipping computation.")
                continue
            save_slice(mixed_layer_depth_slice, save_path)

        print("done.")

        save_path = get_mixed_layer_depth_path(parent_path, delta_sigma)
        # Concatenate all slices into a single dataset
        concatenate_slices(slices_dir, save_path)

        print("All processing complete.")
        client.close()


if __name__ == "__main__":
    compute_mixed_layer_depth("D:/avg", delta_sigma=0.1, time_slice_size=100)
