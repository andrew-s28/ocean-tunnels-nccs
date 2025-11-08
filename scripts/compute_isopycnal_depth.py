"""Computes the depth of a specified isopycnal surface (e.g., sigma_0 = 25.8 kg/m^3) from CROCO model output.

Intended to be used after running `compute_z_levels.py` to ensure z-depths and pressures are available.
"""

import shutil
import warnings
from pathlib import Path

import gsw
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client, LocalCluster
from scipy.optimize import brentq as find_root
from tqdm import tqdm
from zarr.errors import ZarrUserWarning

from utils import get_isopycnal_depth_path, open_grid_with_zdepths, open_model_fields


def _setup_cluster() -> Client:
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="8GiB")
    client = Client(cluster)
    return client


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


def concatenate_slices(parent_path: Path, slices_dir: Path, target_sigma_0: float) -> None:
    """Concatenate all saved isopycnal depth slices into a single dataset and remove the slice files.

    Args:
        parent_path (Path): Parent directory to save the concatenated dataset.
        slices_dir (Path): Directory containing the saved isopycnal depth slices.
        target_sigma_0 (float): The target sigma_0 value used in the slice filenames.

    """
    print("Concatenating all slices into a single dataset...", end="", flush=True)
    slice_files = list(slices_dir.glob("isopycnal_depth_slice_*.zarr"))
    slice_files.sort()
    isopycnal_depth = xr.concat(
        [xr.open_zarr(f) for f in slice_files],
        dim="time",
        compat="no_conflicts",
        coords="minimal",
    )["depth"]
    isopycnal_depth_path = get_isopycnal_depth_path(parent_path, target_sigma_0)
    isopycnal_depth.to_zarr(isopycnal_depth_path)
    print("done.")

    print("Cleaning up slice files...", end="", flush=True)
    [shutil.rmtree(f) for f in slice_files]
    print("done.")


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


def save_isopycnal_depth_slice(
    isopycnal_depth_slice: xr.DataArray,
    save_path: Path,
) -> None:
    """Save a computed isopycnal depth slice to a zarr file.

    Args:
        isopycnal_depth_slice (xr.DataArray): DataArray containing the computed isopycnal depths for the slice.
        save_path (Path): Path to save the zarr file.

    """
    with ProgressBar(), warnings.catch_warnings():
        msg = "Consolidated metadata is currently not part in the Zarr format 3 specification"
        warnings.filterwarnings("ignore", category=ZarrUserWarning, message=msg)
        msg = "Sending large graph of size"
        warnings.filterwarnings("ignore", category=UserWarning, message=msg)
        isopycnal_depth_slice.to_zarr(save_path)


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

    with _setup_cluster() as client:
        print(f"Dask cluster setup. Dashboard link: {client.dashboard_link}")

        print("Opening model fields and grid...", end="", flush=True)
        # Open model fields and grid with z-depths computed
        ds = open_model_fields(parent_path)
        grid = open_grid_with_zdepths(parent_path)
        print("done.")

        # Find the number of time slices needed
        number_of_slices = int(np.ceil(len(ds["time"]) / time_slice_size))

        print("Computing and saving isopycnal depth slices...", end="", flush=True)

        # Process each chunk of times separately
        for i in tqdm(range(number_of_slices), desc="Computing isopycnal depth slices"):
            start_idx = i * time_slice_size
            end_idx = min((i + 1) * time_slice_size, len(ds["time"]))
            ds_slice = ds.isel(time=slice(start_idx, end_idx))

            isopycnal_depth_slice = process_chunk(ds_slice, grid, target_sigma_0)

            # Save the computed isopycnal depths for this slice to a separate NetCDF file, ~1GB total per slice
            save_path = slices_dir / f"isopycnal_depth_slice_{i + 1}.zarr"
            if save_path.exists():
                print(f"File {save_path} already exists. Skipping computation.")
                continue
            save_isopycnal_depth_slice(isopycnal_depth_slice, save_path)

        print("done.")

        concatenate_slices(parent_path, slices_dir, target_sigma_0)

        print("All processing complete.")
        client.close()


if __name__ == "__main__":
    compute_isopycnal_depth(parent_path="D:/avg", target_sigma_0=25.8, time_slice_size=100)
