# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gsw",
#     "numpy",
#     "scipy",
#     "tqdm",
#     "typer",
#     "xarray[accel,io,parallel]",
#     "zarr",
# ]
# ///
"""Module for computing depths on specific density levels from Oceanic Pathways CROCO model output."""

import shutil
import warnings
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path

import dask.array as da
import gsw
import numpy as np
import typer
import xarray as xr
from scipy.optimize import brentq as find_root
from tqdm import tqdm
from zarr.errors import ZarrUserWarning

from utils import (
    compute_sigma_0,
    setup_client,
    setup_cluster,
)


class BaseDepth(ABC):
    """Base class for depth calculations from CROCO model output."""

    def __init__(
        self,
        save_dir: Path | str,
        load_dir: Path | str,
        time_slice_size: int = 100,
    ) -> None:
        """Initialize the BaseDepth class.

        Args:
            save_dir (Path | str): The directory to save the computed depths.
            load_dir (Path | str): The directory to load the model fields from.
            time_slice_size (int): Number of time steps to process in each slice. Default is 100.

        """
        self.save_dir = Path(save_dir)
        self.load_dir = Path(load_dir)
        self.save_path: Path
        self.slices_dir: Path
        self.monthly_mean_path: Path
        self.slice_size = time_slice_size

    def save_slice(
        self,
        slice_da: xr.DataArray,
        index: int,
    ) -> None:
        """Save a computed isopycnal depth slice to a zarr file.

        Args:
            slice_da (xr.DataArray): DataArray containing the computed depths for the slice.
            index (int): Index of the slice to save.

        """
        save_path = self.slices_dir / f"slice_{index}.zarr"
        # Save the computed isopycnal depths for this slice to a separate NetCDF file, ~1GB total per slice
        if save_path.exists():
            tqdm.write(f"File {save_path} already exists. Skipping computation.")
            return
        with warnings.catch_warnings():
            msg = "Consolidated metadata is currently not part in the Zarr format 3 specification"
            warnings.filterwarnings("ignore", category=ZarrUserWarning, message=msg)
            msg = "Sending large graph of size"
            warnings.filterwarnings("ignore", category=UserWarning, message=msg)
            slice_da.to_zarr(save_path)

    def slice_dataset(
        self,
        ds: xr.Dataset,
    ) -> list[xr.Dataset]:
        """Slice a dataset into smaller datasets along the time dimension.

        Args:
            ds (xr.Dataset): The input dataset to be sliced.

        Returns:
            list[xr.Dataset]: A list of sliced datasets.

        """
        # Find the number of time slices needed
        number_of_slices = int(np.ceil(len(ds["time"]) / self.slice_size))
        slices = [
            ds.isel(time=slice(i * self.slice_size, min((i + 1) * self.slice_size, len(ds.time))))
            for i in range(number_of_slices)
        ]
        return slices

    def concatenate_slices(self) -> None:
        """Concatenate all saved slices into a single dataset and remove the slice files.

        Args:
            slices_dir (Path): Directory containing the saved slices.

        """
        typer.echo("Concatenating all slices into a single dataset...", nl=False)
        slice_files = list(self.slices_dir.glob("*.zarr"))
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
            ds.to_zarr(self.save_path)
        typer.echo("done.")

        typer.echo("Cleaning up slice files...", nl=False)
        [shutil.rmtree(f) for f in slice_files]
        self.slices_dir.rmdir()
        typer.echo("done.")

    def open_model_fields(self) -> xr.Dataset:
        """Open Oceanic Pathways model fields from a parent directory.

        Returns:
            xr.Dataset: An xarray Dataset containing the model fields.

        Raises:
            FileNotFoundError: If no model files are found in the specified directory.

        """
        # Find all files with full depth data (-complete.nc)
        model_files = list(self.load_dir.rglob("*-complete.nc"))

        if not model_files:
            msg = f"No model files found in {self.load_dir}."
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

        # Only need temperature and salinity for density calculations
        ds = ds[["temp", "salt", "hc", "h", "zeta"]]

        return ds

    def open_grid(self) -> xr.Dataset:
        """Open the grid file from a parent directory.

        Returns:
            xr.Dataset: An xarray Dataset containing the grid information.

        Raises:
            FileNotFoundError: If the grid file is not found in the specified directory.

        """
        grid_file = self.load_dir / "croco_grd.nc.1b"

        if not grid_file.exists():
            msg = f"Grid file {grid_file.name} not found in directory {self.load_dir}."
            raise FileNotFoundError(msg)

        ds_grid = xr.open_dataset(grid_file)
        return ds_grid

    def open_grid_with_zdepths(self) -> xr.Dataset:
        """Open the grid file with z-depths from a parent directory.

        Returns:
            xr.Dataset: An xarray Dataset containing the grid information with z-depths.

        Raises:
            FileNotFoundError: If the grid file with z-depths is not found in the specified directory.

        """
        grid_file = self.load_dir / "croco_grd_with_z.nc"

        if not grid_file.exists():
            msg = f"Grid file with z-depths {grid_file.name} not found in directory {self.load_dir}."
            msg += "Please compute z-depths first."
            raise FileNotFoundError(msg)

        ds_grid = xr.open_dataset(grid_file)
        return ds_grid

    def compute_monthly_mean(self) -> None:
        """Compute and save the monthly mean isopycnal depth from saved isopycnal depth slices."""
        # Open the zarr dataset
        ds = xr.open_zarr(self.save_path)

        # Resample to monthly means
        monthly_mean_ds = ds.groupby("time.month").mean(dim="time")

        # Save the monthly mean dataset
        monthly_mean_ds.to_zarr(self.monthly_mean_path)
        typer.echo(f"Monthly mean isopycnal depth saved to {self.monthly_mean_path}.")

    def open(self) -> xr.Dataset:
        """Open the isopycnal depth zarr dataset.

        Requires that the isopycnal depth has already been
        computed and saved with `IsopycnalDepth.compute_isopycnal_depth()`.

        Returns:
            xr.Dataset: An xarray Dataset containing the isopycnal depth data.

        Raises:
            FileNotFoundError: If the isopycnal depth file does not exist.

        """
        if not self.save_path.exists():
            msg = f"Isopycnal depth file {self.save_path} does not exist."
            msg += "Please compute it with `IsopycnalDepth.compute_isopycnal_depth()`."
            raise FileNotFoundError(msg)
        ds_isopycnal_depth = xr.open_zarr(self.save_path)
        return ds_isopycnal_depth

    def open_monthly_mean(self) -> xr.Dataset:
        """Open the monthly mean isopycnal depth zarr dataset.

        Requires that the monthly mean isopycnal depth has already been
        computed and saved with `IsopycnalDepth.compute_monthly_mean_isopycnal_depth()`.

        Returns:
            xr.Dataset: An xarray Dataset containing the monthly mean isopycnal depth data.

        Raises:
            FileNotFoundError: If the monthly mean isopycnal depth file does not exist.

        """
        if not self.monthly_mean_path.exists():
            msg = f"Monthly mean isopycnal depth file {self.monthly_mean_path} does not exist."
            msg += "Please compute it with `IsopycnalDepth.compute_monthly_mean_isopycnal_depth()`."
            raise FileNotFoundError(msg)
        ds_monthly_mean = xr.open_zarr(self.monthly_mean_path)
        return ds_monthly_mean

    @abstractmethod
    def compute(self) -> None:
        """Abstract method to compute depths from CROCO model output."""
        raise NotImplementedError

    @abstractmethod
    def _process_chunk(self, ds_chunk: xr.Dataset) -> xr.DataArray:
        """Abstract method to process a chunk of the dataset to compute depths.

        Args:
            ds_chunk (xr.Dataset): Chunk of the model dataset.

        Returns:
            xr.DataArray: DataArray containing the computed depths for the chunk.

        """
        raise NotImplementedError

    @abstractmethod
    def _interpolate(self, z: np.ndarray, sigma_0: np.ndarray, _depth: np.ndarray | None = None) -> float:
        """Abstract method to interpolate to find the depth at which the given sigma0 matches the target sigma0.

        Args:
            z (np.ndarray): Array of depths.
            sigma_0 (np.ndarray): Array of sigma_0 values corresponding to the depths.
            _depth (np.ndarray | None): Array of depths to interpolate to, if provided.

        Returns:
            float: The depth at which sigma_0 matches target_sigma_0, or np.nan if not found.

        """
        raise NotImplementedError


class IsopycnalDepth(BaseDepth):
    """Class to compute isopycnal depths from CROCO model output."""

    def __init__(self, target_sigma_0: float, save_dir: Path | str, load_dir: Path | str) -> None:
        """Initialize the IsopycnalDepth class.

        Args:
            target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.
            save_dir (Path | str): The directory to save the computed isopycnal depths.
            load_dir (Path | str): The directory to load the model fields from.

        """
        self.target_sigma_0 = target_sigma_0
        super().__init__(save_dir, load_dir)
        self.save_path = self.save_dir / f"isopycnal_depth_sigma_{target_sigma_0}.zarr"
        self.slices_dir = self.save_path.with_name("isopycnal_depth_slices")
        self.monthly_mean_path = self.save_dir / f"monthly_mean_{self.save_path.name}"

    def _interpolate(self, z: np.ndarray, sigma_0: np.ndarray, _depth: np.ndarray | None = None) -> float:
        """Interpolate to find the depth at which the given sigma0 matches the target sigma0.

        Intended to be used with the xarray.apply_ufunc defined within this script.

        Args:
            z (np.ndarray): Array of depths.
            sigma_0 (np.ndarray): Array of sigma_0 values corresponding to the depths.

        Returns:
            float: The depth at which sigma_0 matches target_sigma_0, or np.nan if not found.

        """
        try:
            root = find_root(
                lambda depth: np.interp(depth, z, sigma_0, left=np.nan, right=np.nan) - self.target_sigma_0,
                np.min(z),
                np.max(z),
            )
        except ValueError:
            root = np.nan
        # Satisfy type checker, find_root should always return a float if no exception is raised
        if type(root) is float:
            return root
        return np.nan

    def _process_chunk(self, ds_chunk: xr.Dataset) -> xr.DataArray:
        """Process a chunk of the dataset to compute isopycnal depths.

        Args:
            ds_chunk (xr.Dataset): Chunk of the model dataset.

        Returns:
            xr.DataArray: DataArray containing the computed isopycnal depths for the chunk.

        """
        ds_chunk = compute_sigma_0(ds_chunk, self.grid)

        isopycnal_depth_chunk: xr.DataArray = xr.apply_ufunc(
            self._interpolate,
            self.grid["z_rho"],
            ds_chunk["sigma0"],
            input_core_dims=[["s_rho"], ["s_rho"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.grid["z_rho"].dtype],
        )
        isopycnal_depth_chunk = isopycnal_depth_chunk.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1})
        isopycnal_depth_chunk = isopycnal_depth_chunk.rename("depth")

        return isopycnal_depth_chunk

    def compute(self) -> None:
        """Compute the depth of a specified isopycnal surface (e.g., sigma_0 = 25.8 kg/m^3) from CROCO model output.

        Takes about 20 hours to run on a home desktop and saves ~20 GB of data
        when used on the Oceanic Pathways model output.

        """
        self.save_dir.mkdir(exist_ok=True)
        self.slices_dir.mkdir(exist_ok=True)

        with setup_client(setup_cluster(n_workers=4, threads_per_worker=2, memory_limit="8GiB")) as client:
            typer.echo(f"Dask cluster setup. Dashboard link: {client.dashboard_link}")
            typer.echo(f"Loading model files from {self.load_dir}...", nl=False)
            self.grid = self.open_grid_with_zdepths()
            self.model = self.open_model_fields()
            typer.echo("done.")

            # Find the number of time slices needed
            dataset_slices = self.slice_dataset(self.model)

            typer.echo("Computing and saving isopycnal depth slices...", nl=False)

            # Process each chunk of times separately
            for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing isopycnal depth slices")):
                isopycnal_depth_slice = self._process_chunk(ds_slice)
                self.save_slice(isopycnal_depth_slice, i)

            typer.echo("done.")

            self.concatenate_slices()

            typer.echo("All processing complete.")

            client.close()


class MixedLayerDepth(BaseDepth):
    """Class to compute mixed layer depths from CROCO model output."""

    def __init__(self, delta_sigma_0: float, save_dir: Path | str, load_dir: Path | str) -> None:
        """Initialize the MixedLayerDepth class.

        Args:
            delta_sigma_0 (float): The sigma_0 threshold for mixed layer depth calculation.
            save_dir (Path | str): The directory to save the computed mixed layer depths.
            load_dir (Path | str): The directory to load the model fields from.

        """
        self.delta_sigma_0 = delta_sigma_0
        super().__init__(save_dir, load_dir)
        self.save_path = self.save_dir / f"mixed_layer_depth_delta_sigma_{self.delta_sigma_0}.zarr"
        self.slices_dir = self.save_path.with_name("mixed_layer_depth_slices")
        self.monthly_mean_path = self.save_dir / f"monthly_mean_{self.save_path.name}"

    def _interpolate(self, z: np.ndarray, sigma_0: np.ndarray, _depth: np.ndarray | None = None) -> float:
        """Interpolate to find the depth at which the given sigma0 matches the target sigma0.

        Intended to be used with the xarray.apply_ufunc defined within this script.

        Args:
            z (np.ndarray): Array of depths.
            sigma_0 (np.ndarray): Array of sigma_0 values corresponding to the depths.

        Returns:
            float: The depth at which sigma_0 matches surface_sigma_0, or np.nan if not found.

        """
        try:
            root = find_root(
                lambda depth: np.interp(depth, z, sigma_0, left=np.nan, right=np.nan) - self.threshold_sigma_0,
                np.min(z),
                np.max(z),
            )
        except ValueError:
            root = np.nan
        # Satisfy type checker, find_root should always return a float if no exception is raised
        if type(root) is float:
            return root
        return np.nan

    def _process_chunk(self, ds_chunk: xr.Dataset) -> xr.DataArray:
        """Process a chunk of the dataset to compute isopycnal depths.

        Args:
            ds_chunk (xr.Dataset): Chunk of the model dataset.

        Returns:
            xr.DataArray: DataArray containing the computed isopycnal depths for the chunk.

        """
        ds_chunk = compute_sigma_0(ds_chunk, self.grid)
        # grid is from bottom to top so select last index
        self.threshold_sigma_0 = ds_chunk["sigma0"].isel(s_rho=-1) + self.delta_sigma_0

        mixed_layer_depth_chunk: xr.DataArray = xr.apply_ufunc(
            self._interpolate,
            self.grid["z_rho"],
            ds_chunk["sigma0"],
            ds_chunk["threshold_sigma_0"],
            input_core_dims=[["s_rho"], ["s_rho"], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.grid["z_rho"].dtype],
        )
        mixed_layer_depth_chunk = mixed_layer_depth_chunk.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1})
        mixed_layer_depth_chunk = mixed_layer_depth_chunk.rename("depth")

        return mixed_layer_depth_chunk

    def compute(self) -> None:
        """Compute and save the mixed layer depth from CROCO model output using the threshold method."""
        self.save_dir.mkdir(exist_ok=True)
        self.slices_dir.mkdir(exist_ok=True)

        with setup_client(setup_cluster(n_workers=4, threads_per_worker=2, memory_limit="8GiB")) as client:
            typer.echo(f"Dask cluster setup. Dashboard link: {client.dashboard_link}")

            typer.echo(f"Loading model files from {self.load_dir}...", nl=False)
            self.grid = self.open_grid_with_zdepths()
            self.model = self.open_model_fields()
            typer.echo("done.")

            # Slice the dataset into manageable time chunks
            dataset_slices = self.slice_dataset(self.model)

            typer.echo("Computing and saving mixed layer depth slices...", nl=False)

            # Process each chunk of times separately
            for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing mixed layer depth slices")):
                mixed_layer_depth_slice = self._process_chunk(ds_slice)
                self.save_slice(mixed_layer_depth_slice, i)

            typer.echo("done.")

            # Concatenate all slices into a single dataset
            self.concatenate_slices()

            typer.echo("All processing complete.")

            client.close()


class GridWithDepths(BaseDepth):
    """Class to compute z-depths and pressures for CROCO model output."""

    class VerticalCoordinate(Enum):
        """Enumeration for calculating vartical coordinate of w or rho points."""

        RHO = auto()
        W = auto()

    def __init__(self, save_dir: Path | str, load_dir: Path | str) -> None:
        """Initialize the GridWithDepths class.

        Args:
            save_dir (Path | str): The directory to save the grid file with z-depths.
            load_dir (Path | str): The directory to load the model fields from.

        """
        super().__init__(save_dir, load_dir)
        self.grid = self.open_grid()
        self.save_path = Path(save_dir) / "croco_grd_with_z.nc"

        # Only need the first time slice since using time-mean zeta and only need hc, h, zeta variables
        self.model = self.model.isel(time=0)[["hc", "h", "zeta"]]
        # Load into memory for faster computation
        self.model.load()

    def _compute_vertical_transform(self, vertical_coordinate: VerticalCoordinate) -> np.ndarray:
        """Compute a nonlinear vertical transform for the vertical coordinate of a CROCO model.

        More info found here: https://croco-ocean.gitlabpages.inria.fr/croco_doc/model/model.grid.html

        Args:
            vertical_coordinate (VerticalCoordinate): The vertical coordinate type (W or RHO).

        Returns:
            np.ndarray: An array containing the vertical transform.

        Raises:
            ValueError: If an unknown vertical coordinate is provided.

        """
        # Have to convert everything to numpy arrays since xarray doesn't like the dimension broadcasting here
        hc = self.model["hc"].to_numpy()
        h = self.model["h"].to_numpy()
        if vertical_coordinate == self.VerticalCoordinate.W:
            sc = self.model.attrs["sc_w"].to_numpy()
            cs = self.model.attrs["Cs_w"].to_numpy()
        elif vertical_coordinate == self.VerticalCoordinate.RHO:
            sc = self.model.attrs["sc_r"].to_numpy()
            cs = self.model.attrs["Cs_r"].to_numpy()
        else:
            msg = f"Unknown vertical coordinate: {vertical_coordinate}"
            raise ValueError(msg)

        z_0 = (hc * sc + np.multiply.outer(h, cs)) / (hc + h[:, :, np.newaxis])

        return z_0

    def _compute_depth_from_vertical_transform(self, z_0: np.ndarray) -> np.ndarray:
        """Compute the depths of sigma levels from the vertical transform.

        Args:
            z_0 (np.ndarray): The vertical transform array.

        Returns:
            np.ndarray: An array containing the depths of sigma levels.

        """
        # Using time mean zeta since this was the only thing included in model output, but time-varying would be better
        z = (
            self.model["zeta"].to_numpy()[:, :, np.newaxis]
            + (self.model["zeta"].to_numpy()[:, :, np.newaxis] + self.model["h"].to_numpy()[:, :, np.newaxis]) * z_0
        )

        return z

    def _add_to_dataset(self, z_rho: np.ndarray, z_w: np.ndarray) -> xr.Dataset:
        """Compute z-depths and pressures and add them to CROCO grid dataset.

        Args:
            z_rho (np.ndarray): The z-depths at rho points.
            z_w (np.ndarray): The z-depths at w points.

        Returns:
            xr.Dataset: The updated CROCO grid dataset with z-depths and pressures added.

        """
        # Z-depths at rho points
        self.grid["z_rho"] = (["eta_rho", "xi_rho", "s_rho"], z_rho)
        self.grid["z_rho"].attrs["long_name"] = "depth of rho points"
        self.grid["z_rho"].attrs["units"] = "m"

        # Z-depths at w points
        self.grid["z_w"] = (["eta_rho", "xi_rho", "s_w"], z_w)
        self.grid["z_w"].attrs["long_name"] = "depth of w points"
        self.grid["z_w"].attrs["units"] = "m"

        # Pressure at rho points
        self.grid["p_rho"] = (
            ["eta_rho", "xi_rho", "s_rho"],
            gsw.conversions.p_from_z(self.grid["z_rho"], self.grid["lat_rho"]).to_numpy(),
        )
        self.grid["p_rho"].attrs["long_name"] = "sea pressure at rho points"
        self.grid["p_rho"].attrs["units"] = "dbar"

        # Pressure at w points
        self.grid["p_w"] = (
            ["eta_rho", "xi_rho", "s_w"],
            gsw.conversions.p_from_z(self.grid["z_w"], self.grid["lat_rho"]).to_numpy(),
        )
        self.grid["p_w"].attrs["long_name"] = "sea pressure at w points"
        self.grid["p_w"].attrs["units"] = "dbar"
        return self.grid

    def compute(self) -> None:
        """Compute z-depths and pressures for a CROCO model and save to a new grid file."""
        z_0 = self._compute_vertical_transform(self.VerticalCoordinate.RHO)
        z_rho = self._compute_depth_from_vertical_transform(z_0)

        z_0 = self._compute_vertical_transform(self.VerticalCoordinate.W)
        z_w = self._compute_depth_from_vertical_transform(z_0)
        ds_grid = self._add_to_dataset(z_rho, z_w)

        ds_grid.to_netcdf(self.save_path)


class DensityAtMixedLayerDepth(BaseDepth):
    """Class to compute density at mixed layer depths from CROCO model output."""

    def __init__(self, mixed_layer_depth: MixedLayerDepth) -> None:
        """Initialize the DensityAtMixedLayerDepth class.

        Args:
            mixed_layer_depth (MixedLayerDepth): An instance of the MixedLayerDepth class.

        Raises:
            FileNotFoundError: If the mixed layer depth file does not exist.

        """
        self.mixed_layer_depth_cls = mixed_layer_depth
        super().__init__(mixed_layer_depth.save_dir, mixed_layer_depth.load_dir, time_slice_size=48)
        self.save_path = self.save_dir / f"density_at_mld_delta_sigma_{mixed_layer_depth.delta_sigma_0}.zarr"
        self.slices_dir = self.save_path.with_name("density_at_mld_slices")
        self.monthly_mean_path = self.save_dir / f"monthly_mean_{self.save_path.name}"

        if not self.mixed_layer_depth_cls.save_path.exists():
            msg = f"Mixed layer depth file {self.mixed_layer_depth_cls.save_path} does not exist."
            msg += "Please compute it with `MixedLayerDepth.compute()`."
            raise FileNotFoundError(msg)

        self.mixed_layer_depth = self.mixed_layer_depth_cls.open()["depth"]

    def _interpolate(self, z: da.Array, sigma_0: xr.DataArray, _depth: da.Array | None = None) -> float:
        """Interpolate to to find sigma_0 at a given depth.

        Intended to be used with the xarray.apply_ufunc defined within this script.

        Args:
            z (np.ndarray): Array of depths.
            sigma_0 (np.ndarray): Array of sigma_0 values corresponding to the depths.
            _depth (np.ndarray): The depth to interpolate to.

        Returns:
            float: The depth at which sigma_0 matches surface_sigma_0, or np.nan if not found.

        """
        print(sigma_0.shape, z.shape, _depth)
        # Skip if all mld is NaN
        if _depth is None:
            return np.nan
        for i in range(sigma_0.shape[0]):
            try:
                root = find_root(
                    lambda depth: np.interp(depth, z, sigma_0[:, i], left=np.nan, right=np.nan) - _depth,
                    np.min(z),
                    np.max(z),
                )
            except ValueError:
                root = np.nan
        return root

    def interp_vertical_decreasing(self, depth, var, zt):
        """depth: (eta, xi, s)
        var:   (eta, xi, s)
        zt:    (eta, xi)
        returns: (eta, xi)

        Linear interpolation, fill_value=np.nan, no extrapolation.
        """
        ny, nx, nz = depth.shape
        ncol = ny * nx

        # reshape for vectorized np.interp
        depth2 = depth.reshape(-1, nz)  # (ncol, nz)
        var2 = var.reshape(-1, nz)  # (ncol, nz)
        zt2 = zt.reshape(-1)  # (ncol,)

        # Indices of first depth greater than target (vectorized search)
        # depth2 is monotonic: bottom->surface (increasing)
        hi = np.argmax(depth2 >= zt2[:, None], axis=1)
        hi = np.clip(hi, 1, nz - 1)
        lo = hi - 1

        # Gather values at bounding indices
        d0 = depth2[np.arange(ncol), lo]
        d1 = depth2[np.arange(ncol), hi]
        v0 = var2[np.arange(ncol), lo]
        v1 = var2[np.arange(ncol), hi]

        # Linear interpolation weights
        w = (zt2 - d0) / (d1 - d0)
        out = v0 + w * (v1 - v0)

        # Mask points outside vertical range
        out[(zt2 < depth2[:, 0]) | (zt2 > depth2[:, -1])] = np.nan

        return out.reshape(ny, nx)

    def interp_block(self, ds):
        depth = ds["z_rho"]  # (eta, xi, s)
        var = ds["sigma_0"]  # (eta, xi, 1, s)
        zt = ds["mld"]  # (eta, xi, 1)

        # extract time index (length 1)
        depth = depth.transpose("eta_rho", "xi_rho", "s_rho").values
        var = var.isel(time=0).transpose("eta_rho", "xi_rho", "s_rho").values
        zt = zt.isel(time=0).transpose("eta_rho", "xi_rho").values

        out = self.interp_vertical_decreasing(depth, var, zt)

        coords = {k: ds.coords[k] for k in ds.coords if k != "s_rho"}  # keep all coords except vertical

        # return DataArray with the same time dimension (size 1)
        return xr.DataArray(
            out[..., np.newaxis],  # add time axis at position 2 (eta,xi,time)
            dims=("eta_rho", "xi_rho", "time"),
            coords=coords,
            attrs=ds.attrs,
        )

    def _process_chunk(self, ds_chunk: xr.Dataset) -> xr.DataArray:
        """Process a chunk of the dataset to compute density at mixed layer depths.

        Args:
            ds_chunk (xr.Dataset): Chunk of the model dataset.

        Returns:
            xr.DataArray: DataArray containing the computed densities at mixed layer depths for the chunk.

        """
        ds_chunk = compute_sigma_0(ds_chunk, self.grid)

        template = xr.zeros_like(
            ds_chunk["mld"],
            dtype=ds_chunk["mld"].dtype,
        )

        density_at_mld_chunk = xr.map_blocks(
            self.interp_block,
            xr.Dataset(
                {
                    "sigma_0": ds_chunk["sigma_0"],
                    "mld": ds_chunk["mld"],
                    "z_rho": self.grid["z_rho"],
                },
            ),
            template=template,
        )

        # density_at_mld_chunk: xr.DataArray = xr.apply_ufunc(
        #     self._interpolate,
        #     self.grid["z_rho"],
        #     ds_chunk["sigma0"],
        #     ds_chunk["mld"],
        #     input_core_dims=[["s_rho"], ["s_rho"], []],
        #     output_core_dims=[[]],
        #     vectorize=True,
        #     dask="allowed",
        #     output_dtypes=[self.grid["z_rho"].dtype],
        # )
        density_at_mld_chunk = density_at_mld_chunk.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1})
        density_at_mld_chunk = density_at_mld_chunk.rename("density")

        return density_at_mld_chunk

    def compute(self) -> None:
        """Compute and save the mixed layer depth from CROCO model output using the threshold method."""
        self.save_dir.mkdir(exist_ok=True)
        self.slices_dir.mkdir(exist_ok=True)

        with setup_client(setup_cluster(n_workers=2, threads_per_worker=2, memory_limit="16GiB")) as client:
            typer.echo(f"Dask cluster setup. Dashboard link: {client.dashboard_link}")

            typer.echo(f"Loading model files from {self.load_dir}...", nl=False)
            self.grid = self.open_grid_with_zdepths()
            self.model = self.open_model_fields()
            typer.echo("done.")

            # Add mixed layer depth to the model dataset for interpolation
            self.model["mld"] = self.mixed_layer_depth

            # Slice the dataset into manageable time chunks
            dataset_slices = self.slice_dataset(self.model)

            typer.echo("Computing and saving mixed layer depth slices...", nl=False)

            # Process each chunk of times separately
            for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing mixed layer depth slices")):
                mixed_layer_depth_slice = self._process_chunk(ds_slice)
                self.save_slice(mixed_layer_depth_slice, i)

            typer.echo("done.")

            # Concatenate all slices into a single dataset
            self.concatenate_slices()

            typer.echo("All processing complete.")

            client.close()


app = typer.Typer(
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
    help="Compute depths from CROCO model output. Modes: mld, isopycnal, mld_density",
)

ALLOWED_MODES = {"mld", "isopycnal", "mld_density"}

SAVE_DIR_OPTION = typer.Option(Path(), "-s", "--save-dir", help="Directory to save results")
LOAD_DIR_OPTION = typer.Option(Path(), "-l", "--load-dir", help="Directory to load model files from")


@app.command()
def main(
    mode: str = typer.Argument(..., help="One of: mld, isopycnal, mld_density"),
    value: float = typer.Argument(..., help="delta_sigma (for mld / mld_density) or target sigma (for isopycnal)"),
    save_dir: Path = SAVE_DIR_OPTION,
    load_dir: Path = LOAD_DIR_OPTION,
) -> None:
    """CLI entry point to compute isopycnal depth, mixed-layer depth, or density at MLD.

    Raises:
        typer.Exit: If an invalid mode is provided.

    """
    mode = mode.lower()
    if mode not in ALLOWED_MODES:
        typer.echo(f"Invalid mode: {mode}. Choose one of {sorted(ALLOWED_MODES)}")
        raise typer.Exit(code=1)

    if mode == "isopycnal":
        iso = IsopycnalDepth(target_sigma_0=float(value), save_dir=save_dir, load_dir=load_dir)
        typer.echo(f"Computing isopycnal depth for sigma0 = {value}")
        iso.compute()

    elif mode == "mld":
        mld = MixedLayerDepth(delta_sigma_0=float(value), save_dir=save_dir, load_dir=load_dir)
        typer.echo(f"Computing mixed layer depth with delta_sigma = {value}")
        mld.compute()

    elif mode == "mld_density":
        # Ensure MLD is available; compute if not present
        mld = MixedLayerDepth(delta_sigma_0=float(value), save_dir=save_dir, load_dir=load_dir)
        if not mld.save_path.exists():
            typer.echo("Mixed layer depth file not found. Would you like to compute it now? [y/n]: ", nl=False)
            user_input = input()
            if user_input.lower() == "y":
                mld.compute()
        # rechunk_for_mld_density(save_dir=save_dir, load_dir=load_dir, delta_sigma_0=float(value))
        density = DensityAtMixedLayerDepth(mld)
        typer.echo(f"Computing density at MLD for delta_sigma = {value}")
        density.compute()

    typer.echo("Done.")


if __name__ == "__main__":
    app()
