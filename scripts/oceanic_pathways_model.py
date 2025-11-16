import warnings
from enum import Enum, auto
from pathlib import Path

import gsw
import numpy as np
import xarray as xr
from scipy.optimize import brentq as find_root
from tqdm import tqdm

from utils import (
    compute_sigma_0,
    concatenate_slices,
    open_grid,
    open_grid_with_zdepths,
    open_model_fields,
    save_slice,
    setup_cluster,
    slice_dataset,
)


class IsopycnalDepth:
    """Class to compute isopycnal depths from CROCO model output."""

    def __init__(self, target_sigma_0: float, save_dir: Path | str, load_dir: Path | str) -> None:
        """Initialize the IsopycnalDepth class.

        Args:
            target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.
            save_dir (Path | str): The directory to save the computed isopycnal depths.
            load_dir (Path | str): The directory to load the model fields from.

        """
        self.target_sigma_0 = target_sigma_0
        self.save_dir = Path(save_dir)
        self.load_dir = Path(load_dir)
        self.grid = open_grid_with_zdepths(self.load_dir)
        self.model = open_model_fields(self.load_dir)
        self.save_path = self.save_dir / f"isopycnal_depth_sigma_{target_sigma_0}.zarr"
        self.monthly_mean_path = self.save_dir / f"monthly_mean_{self.save_path.name}"

    def interpolate_to_density_level(self, z: np.ndarray, sigma_0: np.ndarray, target_sigma_0: float = 25.8) -> float:
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

    def _process_isopycnal_chunk(self, ds_chunk: xr.Dataset, grid: xr.Dataset, target_sigma_0: float) -> xr.DataArray:
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
            self.interpolate_to_density_level,
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

    def compute_isopycnal_depth(self, time_slice_size: int) -> None:
        """Compute the depth of a specified isopycnal surface (e.g., sigma_0 = 25.8 kg/m^3) from CROCO model output.

        Takes about 20 hours to run on a home desktop and saves ~20 GB of data
        when used on the Oceanic Pathways model output.

        Args:
            time_slice_size (int): Number of time steps to process in each slice.

        """
        slices_dir = get_isopycnal_depth_path(self.load_dir, self.target_sigma_0).with_name("isopycnal_depth_slices")
        slices_dir.mkdir(exist_ok=True)

        with setup_cluster(n_workers=4, threads_per_worker=2, memory_limit="8GiB") as client:
            print(f"Dask cluster setup. Dashboard link: {client.dashboard_link}")

            # Find the number of time slices needed
            dataset_slices = slice_dataset(self.model, time_slice_size)

            print("Computing and saving isopycnal depth slices...", end="", flush=True)

            # Process each chunk of times separately
            for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing isopycnal depth slices")):
                isopycnal_depth_slice = self._process_isopycnal_chunk(ds_slice, self.grid, self.target_sigma_0)

                # Save the computed isopycnal depths for this slice to a separate NetCDF file, ~1GB total per slice
                save_path = slices_dir / f"isopycnal_depth_slice_{i + 1}.zarr"
                if save_path.exists():
                    tqdm.write(f"File {save_path} already exists. Skipping computation.")
                    continue
                save_slice(isopycnal_depth_slice, save_path)

            print("done.")

            save_path = get_isopycnal_depth_path(self.save_dir, self.target_sigma_0)
            concatenate_slices(save_path, slices_dir)

            print("All processing complete.")
            client.close()

    def compute_monthly_mean_isopycnal_depth(self) -> None:
        """Compute and save the monthly mean isopycnal depth from saved isopycnal depth slices."""
        isopycnal_depth_path = get_isopycnal_depth_path(self.load_dir, self.target_sigma_0)

        # Open the zarr dataset
        ds = xr.open_zarr(isopycnal_depth_path)

        # Resample to monthly means
        monthly_mean_ds = ds.groupby("time.month").mean(dim="time")

        # Save the monthly mean dataset
        monthly_mean_path = get_monthly_mean_isopycnal_depth_path(self.load_dir, self.target_sigma_0)
        monthly_mean_ds.to_zarr(monthly_mean_path)
        print(f"Monthly mean isopycnal depth saved to {monthly_mean_path}.")


def save_slice(
    slice_da: xr.DataArray,
    save_path: Path,
) -> None:
    """Save a computed isopycnal depth slice to a zarr file.

    Args:
        slice_da (xr.DataArray): DataArray containing the computed depths for the slice.
        save_path (Path): Path to save the zarr file.

    """
    with warnings.catch_warnings():
        msg = "Consolidated metadata is currently not part in the Zarr format 3 specification"
        warnings.filterwarnings("ignore", category=ZarrUserWarning, message=msg)
        msg = "Sending large graph of size"
        warnings.filterwarnings("ignore", category=UserWarning, message=msg)
        slice_da.to_zarr(save_path)


def slice_dataset(
    ds: xr.Dataset,
    slice_size: int,
) -> list[xr.Dataset]:
    """Slice a dataset into smaller datasets along the time dimension.

    Args:
        ds (xr.Dataset): The input dataset to be sliced.
        slice_size (int): The number of time steps in each slice.

    Returns:
        list[xr.Dataset]: A list of sliced datasets.

    """
    # Find the number of time slices needed
    number_of_slices = int(np.ceil(len(ds["time"]) / slice_size))
    slices = [
        ds.isel(time=slice(i * slice_size, min((i + 1) * slice_size, len(ds.time)))) for i in range(number_of_slices)
    ]
    return slices


def concatenate_slices(save_path: Path | str, slices_dir: Path | str) -> None:
    """Concatenate all saved slices into a single dataset and remove the slice files.

    Args:
        save_path (Path | str): Path to save the concatenated dataset.
        slices_dir (Path | str): Directory containing the saved isopycnal depth slices.

    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if isinstance(slices_dir, str):
        slices_dir = Path(slices_dir)

    print("Concatenating all slices into a single dataset...", end="", flush=True)
    slice_files = list(slices_dir.glob("*slice_*.zarr"))
    slice_files.sort()
    ds = xr.concat(
        [xr.open_zarr(f) for f in slice_files],
        dim="time",
        compat="no_conflicts",
        coords="minimal",
    )
    with warnings.catch_warnings():
        msg = "Consolidated metadata is currently not part in the Zarr format 3 specification"
        warnings.filterwarnings("ignore", category=ZarrUserWarning, message=msg)
        msg = "Sending large graph of size"
        warnings.filterwarnings("ignore", category=UserWarning, message=msg)
        ds.to_zarr(save_path)
    print("done.")

    print("Cleaning up slice files...", end="", flush=True)
    [shutil.rmtree(f) for f in slice_files]
    slices_dir.rmdir()
    print("done.")


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
        msg += "Please compute z-depths first."
        raise FileNotFoundError(msg)

    ds_grid = xr.open_dataset(grid_file)
    return ds_grid


def get_mixed_layer_depth_path(parent_path: str | Path, delta_sigma: float) -> Path:
    """Get the path to the mixed layer depth zarr directory.

    Args:
        parent_path (str | Path): Path to the parent directory containing the mixed layer depth data.
        delta_sigma (float): The sigma_0 threshold for mixed layer depth calculation.

    Returns:
        Path: The path to the mixed layer depth zarr directory.

    """
    if isinstance(parent_path, str):
        parent_path = Path(parent_path)

    mld_path = parent_path / f"mixed_layer_depth_delta_sigma_{delta_sigma}.zarr"
    return mld_path


def open_isopycnal_depth(parent_path: str | Path, target_sigma_0: float) -> xr.Dataset:
    """Open the isopycnal depth zarr dataset.

    Args:
        parent_path (str | Path): Path to the parent directory containing the isopycnal depth data.
        target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.

    Returns:
        xr.Dataset: An xarray Dataset containing the isopycnal depth data.

    """
    isopycnal_depth_path = get_isopycnal_depth_path(parent_path, target_sigma_0)
    ds_isopycnal_depth = xr.open_zarr(isopycnal_depth_path)
    return ds_isopycnal_depth


def open_monthly_mean_isopycnal_depth(parent_path: str | Path, target_sigma_0: float) -> xr.Dataset:
    """Open the monthly mean isopycnal depth zarr dataset.

    Args:
        parent_path (str | Path): Path to the parent directory containing the monthly mean isopycnal depth data.
        target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.

    Returns:
        xr.Dataset: An xarray Dataset containing the monthly mean isopycnal depth data.

    """
    monthly_mean_path = get_monthly_mean_isopycnal_depth_path(parent_path, target_sigma_0)

    ds_monthly_mean = xr.open_zarr(monthly_mean_path)
    return ds_monthly_mean


class VerticalCoordinate(Enum):
    """Enumeration for calculating vartical coordinate of w or rho points."""

    RHO = auto()
    W = auto()


def compute_vertical_transform(ds: xr.Dataset, vertical_coordinate: VerticalCoordinate) -> np.ndarray:
    """Compute a nonlinear vertical transform for the vertical coordinate of a CROCO model.

    More info found here: https://croco-ocean.gitlabpages.inria.fr/croco_doc/model/model.grid.html

    Args:
        ds (xr.Dataset): The model dataset containing necessary variables.
        vertical_coordinate (VerticalCoordinate): The vertical coordinate type (W or RHO).

    Returns:
        np.ndarray: An array containing the vertical transform.

    Raises:
        ValueError: If an unknown vertical coordinate is provided.

    """
    # Have to convert everything to numpy arrays since xarray doesn't like the dimension broadcasting here
    hc = ds["hc"].to_numpy()
    h = ds["h"].to_numpy()
    if vertical_coordinate == VerticalCoordinate.W:
        sc = ds.attrs["sc_w"]
        cs = ds.attrs["Cs_w"]
    elif vertical_coordinate == VerticalCoordinate.RHO:
        sc = ds.attrs["sc_r"]
        cs = ds.attrs["Cs_r"]
    else:
        msg = f"Unknown vertical coordinate: {vertical_coordinate}"
        raise ValueError(msg)

    z_0 = (hc * sc + np.multiply.outer(h, cs)) / (hc + h[:, :, np.newaxis])

    return z_0


def compute_depths(ds: xr.Dataset, z0: np.ndarray) -> np.ndarray:
    """Compute the depths of sigma levels from the vertical transform.

    Args:
        ds (xr.Dataset): The model dataset containing necessary variables.
        z0 (np.ndarray): The vertical transform array.

    Returns:
        np.ndarray: An array containing the depths of sigma levels.

    """
    # Using time mean zeta since this was the only thing included in model output, but time-varying would be better
    z = (
        ds["zeta"].to_numpy()[:, :, np.newaxis]
        + (ds["zeta"].to_numpy()[:, :, np.newaxis] + ds["h"].to_numpy()[:, :, np.newaxis]) * z0
    )

    return z


def setup_dataset(z_rho: np.ndarray, z_w: np.ndarray, ds_grid: xr.Dataset) -> xr.Dataset:
    """Update the CROCO grid dataset with z-depths and pressures.

    Args:
        z_rho (np.ndarray): The z-depths at rho points.
        z_w (np.ndarray): The z-depths at w points.
        ds_grid (xr.Dataset): The CROCO grid dataset.

    Returns:
        xr.Dataset: The updated CROCO grid dataset with z-depths and pressures added.

    """
    # Z-depths at rho points
    ds_grid["z_rho"] = (["eta_rho", "xi_rho", "s_rho"], z_rho)
    ds_grid["z_rho"].attrs["long_name"] = "depth of rho points"
    ds_grid["z_rho"].attrs["units"] = "m"

    # Z-depths at w points
    ds_grid["z_w"] = (["eta_rho", "xi_rho", "s_w"], z_w)
    ds_grid["z_w"].attrs["long_name"] = "depth of w points"
    ds_grid["z_w"].attrs["units"] = "m"

    # Pressure at rho points
    ds_grid["p_rho"] = (
        ["eta_rho", "xi_rho", "s_rho"],
        gsw.conversions.p_from_z(ds_grid["z_rho"], ds_grid["lat_rho"]).to_numpy(),
    )
    ds_grid["p_rho"].attrs["long_name"] = "sea pressure at rho points"
    ds_grid["p_rho"].attrs["units"] = "dbar"

    # Pressure at w points
    ds_grid["p_w"] = (
        ["eta_rho", "xi_rho", "s_w"],
        gsw.conversions.p_from_z(ds_grid["z_w"], ds_grid["lat_rho"]).to_numpy(),
    )
    ds_grid["p_w"].attrs["long_name"] = "sea pressure at w points"
    ds_grid["p_w"].attrs["units"] = "dbar"
    return ds_grid


def compute_depth_and_pressure(parent_path: str) -> None:
    """Compute z-depths and pressures for a CROCO model and save to a new grid file.

    Args:
        parent_path (str): Path to the parent directory where the new grid file will be saved.

    """
    ds_model = open_model_fields(parent_path)
    ds_grid = open_grid(parent_path)

    # Only need the first time slice since using time-mean zeta and only need hc, h, zeta variables
    ds_slice = ds_model.isel(time=0)[["hc", "h", "zeta"]]
    # Load into memory for faster computation
    ds_slice.load()

    z_0 = compute_vertical_transform(ds_slice, VerticalCoordinate.RHO)
    z_rho = compute_depths(ds_slice, z_0)

    z_0 = compute_vertical_transform(ds_slice, VerticalCoordinate.W)
    z_w = compute_depths(ds_slice, z_0)

    ds_grid = setup_dataset(z_rho, z_w, ds_grid)

    ds_grid.to_netcdf(Path(parent_path) / "croco_grd_with_z.nc")


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


def _process_mld_chunk(ds_chunk: xr.Dataset, grid: xr.Dataset, delta_sigma_0: float) -> xr.DataArray:
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

    slices_dir = get_mixed_layer_depth_path(parent_path, delta_sigma).with_name(
        f"mixed_layer_depth_delta_sigma_{delta_sigma}",
    )
    slices_dir.mkdir(exist_ok=True)

    with setup_cluster(n_workers=4, threads_per_worker=2, memory_limit="8GiB") as client:
        print(f"Dask cluster setup. Dashboard link: {client.dashboard_link}")

        print("Opening model fields and grid...", end="", flush=True)
        # Open model fields and grid with z-depths computed
        ds = open_model_fields(parent_path)
        grid = open_grid_with_zdepths(parent_path)
        print("done.")

        # Slice the dataset into manageable time chunks
        dataset_slices = slice_dataset(ds, time_slice_size)

        print("Computing and saving mixed layer depth slices...", end="", flush=True)

        # Process each chunk of times separately
        for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing mixed layer depth slices")):
            mixed_layer_depth_slice = _process_mld_chunk(ds_slice, grid, delta_sigma)

            # Save the computed mixed layer depths for this slice to a separate zarr file
            save_path = slices_dir / f"mixed_layer_depth_slice_{i + 1}.zarr"
            if save_path.exists():
                tqdm.write(f"File {save_path} already exists. Skipping computation.")
                continue
            save_slice(mixed_layer_depth_slice, save_path)

        print("done.")

        save_path = get_mixed_layer_depth_path(parent_path, delta_sigma)
        # Concatenate all slices into a single dataset
        concatenate_slices(save_path, slices_dir)

        print("All processing complete.")
        client.close()
