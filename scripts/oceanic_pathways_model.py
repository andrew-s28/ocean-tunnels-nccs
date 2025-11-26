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
from datetime import UTC, datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import gsw
import numpy as np
import typer
import xarray as xr
from tqdm import tqdm
from zarr.errors import ZarrUserWarning

from utils import (
    compute_sigma_0,
    setup_client,
)

# Ignore all-NaN slice warnings globally for interpolations
# Context manager doesn't seem to play nice with dask/xarray apply_ufunc parallelism
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"All-NaN slice encountered")

AUTHOR_ATTRIBUTES = {
    "creator_name": "Andrew Scherer",
    "creator_email": "scherand@oregonstate.edu",
    "creator_institution": "College of Earth, Ocean, and Atmospheric Sciences, Oregon State University",
}


class BaseModel(ABC):
    """Base class for loading data and slicing datasets from CROCO model output."""

    def __init__(  # noqa: PLR0913
        self,
        save_dir: Path | str,
        load_dir: Path | str,
        time_slice_size: int = 100,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "8GB",
    ) -> None:
        """Initialize the BaseModel class.

        Args:
            save_dir (Path | str): The directory to save the computed depths.
            load_dir (Path | str): The directory to load the model fields from.
            time_slice_size (int): Number of time steps to process in each slice. Default is 100.
            n_workers (int): Number of Dask workers to use. Default is 4.
            threads_per_worker (int): Number of threads per Dask worker. Default is 2.
            memory_limit (str): Memory limit per Dask worker. Default is "8GB".

        """
        self.save_dir = Path(save_dir)
        self.load_dir = Path(load_dir)
        self.save_path: Path
        self.slices_dir: Path
        self.monthly_mean_path: Path
        self.slice_size = time_slice_size
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit

    def _save_slice(
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

    def _slice_dataset(
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

    def _concatenate_slices(self) -> xr.Dataset:
        """Concatenate all saved slices into a single dataset and remove the slice files.

        Args:
            slices_dir (Path): Directory containing the saved slices.

        Returns:
            xr.Dataset: The concatenated dataset.

        """
        slice_files = list(self.slices_dir.glob("*.zarr"))
        slice_files.sort()
        ds = xr.concat(
            [xr.open_zarr(f) for f in slice_files],
            dim="time",
            compat="no_conflicts",
            coords="minimal",
        )

        return ds

    def _save_dataset(self, ds: xr.Dataset) -> None:
        """Save the concatenated dataset to a zarr store and remove the slice files.

        Args:
            ds (xr.Dataset): The dataset to save.

        """
        with warnings.catch_warnings():
            msg = "Consolidated metadata is currently not part in the Zarr format 3 specification"
            warnings.filterwarnings("ignore", category=ZarrUserWarning, message=msg)
            msg = "Sending large graph of size"
            warnings.filterwarnings("ignore", category=UserWarning, message=msg)
            ds.to_zarr(self.save_path)

        slice_files = list(self.slices_dir.glob("*.zarr"))
        [shutil.rmtree(f) for f in slice_files]
        self.slices_dir.rmdir()

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

    def open(self) -> xr.Dataset:
        """Open a zarr store as an xarray Dataset.

        Requires that the computation has been saved by calling the `compute()` method.

        Returns:
            xr.Dataset: An xarray Dataset containing the data.

        Raises:
            FileNotFoundError: If the file does not exist.

        """
        if not self.save_path.exists():
            msg = f"File {self.save_path} does not exist. Please compute it with `compute()`."
            raise FileNotFoundError(msg)
        ds = xr.open_zarr(self.save_path)
        return ds

    @abstractmethod
    def compute(self) -> None:
        """Abstract method to run interpolation computation over CROCO model data."""
        raise NotImplementedError


class GridWithDepths(BaseModel):
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


class BaseInterpolation(BaseModel, ABC):
    """Base class for interpolating to depths and/or densities from CROCO model output."""

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
        super().__init__(save_dir, load_dir, time_slice_size)

    @abstractmethod
    def _process_slice(self, ds_slice: xr.Dataset) -> xr.DataArray:
        """Abstract method to process a slice of the dataset to compute depths.

        Args:
            ds_slice (xr.Dataset): Slice of the model dataset.

        Returns:
            xr.DataArray: DataArray containing the computed depths for the slice.

        """
        raise NotImplementedError

    @abstractmethod
    def _generate_attributes(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Abstract method to generate variable and global attributes for the output dataset.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: A tuple containing variable attributes and global attributes.

        """
        raise NotImplementedError

    @staticmethod
    def _interpolate_to_density(depth: np.ndarray, density: np.ndarray, target_density: np.ndarray) -> np.ndarray:
        """Vectorized interpolation to find the depth of a target density level from *decreasing* density values.

        Args:
            depth (np.ndarray): 3D array of depths with shape (eta, xi, s).
            density (np.ndarray): 3D array of vertically *decreasing* density values with shape (eta, xi, s).
            target_density (np.ndarray): 2D array of target density values with shape (eta, xi).

        Returns:
            np.ndarray: 2D array of interpolated variable values at target depths with shape (eta, xi).

        """
        ny, nx, nz = depth.shape
        nxy = nx * ny

        # reshape for vectorized interpolation
        depth = depth.reshape(-1, nz)  # (nxy, nz)
        density = density.reshape(-1, nz)  # (nxy, nz)
        target_density = target_density.reshape(-1)  # (nxy,)

        # Indices of first density less than target, since argmax finds first True
        # density must be decreasing from bottom->surface
        hi = np.argmax(density <= target_density[:, None], axis=1)
        hi = np.clip(hi, 1, nz - 1)
        lo = hi - 1

        # Get values at bounding indices
        d0 = depth[np.arange(nxy), lo]
        d1 = depth[np.arange(nxy), hi]
        v0 = density[np.arange(nxy), lo]
        v1 = density[np.arange(nxy), hi]

        # Slope is rise over run
        slope = (d1 - d0) / (v1 - v0)
        out = d0 + slope * (target_density - v0)

        # Mask points where density is all NaN
        out[np.all(np.isnan(density), axis=1)] = np.nan

        # Now mask points outside density range
        out[(target_density < np.nanmin(density, axis=1)) | (target_density > np.nanmax(density, axis=1))] = np.nan

        return out.reshape(ny, nx)

    @staticmethod
    def _interpolate_to_depth(depth: np.ndarray, var: np.ndarray, target_depth: np.ndarray) -> np.ndarray:
        """Vectorized interpolation to find the value of var at target_depth for array with *increasing* depth values.

        Args:
            depth (np.ndarray): 3D array of *increasing* depths with shape (eta, xi, s).
            var (np.ndarray): 3D array of variable values with shape (eta, xi, s).
            target_depth (np.ndarray): 2D array of target depths with shape (eta, xi).

        Returns:
            np.ndarray: 2D array of interpolated variable values at target depths with shape (eta, xi).

        """
        ny, nx, nz = depth.shape
        nxy = nx * ny

        # reshape for vectorized interpolation
        depth = depth.reshape(-1, nz)  # (nxy, nz)
        var = var.reshape(-1, nz)  # (nxy, nz)
        target_depth = target_depth.reshape(-1)  # (nxy,)

        # Indices of first depth greater than target, since argmax finds first True
        # depth must be monotonic increasing from bottom->surface
        hi = np.argmax(depth >= target_depth[:, None], axis=1)
        hi = np.clip(hi, 1, nz - 1)
        lo = hi - 1

        # Get values at bounding indices
        d0 = depth[np.arange(nxy), lo]
        d1 = depth[np.arange(nxy), hi]
        v0 = var[np.arange(nxy), lo]
        v1 = var[np.arange(nxy), hi]

        # Slope is rise over run
        slope = (v1 - v0) / (d1 - d0)
        out = v0 + slope * (target_depth - d0)

        # Mask points where var is all NaN
        # This shouldn't be the case for depth but just in case and to match density interpolation
        out[np.all(np.isnan(depth), axis=1)] = np.nan

        # Now mask points outside vertical range
        out[(target_depth < depth[:, 0]) | (target_depth > depth[:, -1])] = np.nan

        return out.reshape(ny, nx)

    @staticmethod
    def _assign_attributes(
        ds: xr.Dataset,
        variable_attributes: dict[str, Any] | None = None,
        global_attributes: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        """Add attributes to the output dataset.

        Args:
            ds (xr.Dataset): The dataset to add attributes to.
            variable_attributes (dict[str, Any]): Variable-specific attributes to add to the dataset.
            global_attributes (dict[str, Any]): Global attributes to add to the dataset.

        Returns:
            xr.Dataset: The dataset with added attributes.

        """
        if variable_attributes is not None:
            for var_name, attrs in variable_attributes.items():
                if var_name in ds:
                    ds[var_name].attrs.update(attrs)
        if global_attributes is not None:
            ds.attrs.update(global_attributes)
        return ds


class IsopycnalDepth(BaseInterpolation):
    """Class to compute isopycnal depths from CROCO model output."""

    def __init__(self, target_sigma_0: float, save_dir: Path | str, load_dir: Path | str) -> None:
        """Initialize the IsopycnalDepth class.

        Args:
            target_sigma_0 (float): The target sigma_0 value for the isopycnal depth.
            save_dir (Path | str): The directory to save the computed isopycnal depths.
            load_dir (Path | str): The directory to load the model fields from.

        """
        self.target_sigma_0 = target_sigma_0
        super().__init__(save_dir, load_dir, time_slice_size=12 * 12)
        self.save_path = self.save_dir / f"isopycnal_depth_sigma_{target_sigma_0}.zarr"
        self.slices_dir = self.save_path.with_name("isopycnal_depth_slices")
        self.monthly_mean_path = self.save_dir / f"monthly_mean_{self.save_path.name}"

    def _generate_attributes(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate variable and global attributes for the isopycnal depth dataset.

        Returns:
            dict[str, Any]: A dictionary containing variable and global attributes.

        """
        variable_attributes = {str(var): self.model[var].attrs for var in self.model.sizes}

        variable_attributes.update(
            {
                "depth": {
                    "long_name": f"depth of isopycnal surface where sigma_0={self.target_sigma_0} kg/m^3",
                    "standard_name": "depth_of_isosurface_of_sea_water_sigma_theta",
                    "units": "m",
                },
                "sigma_theta": {
                    "long_name": "isopycnal surface potential density referenced to the surface",
                    "standard_name": "sea_water_sigma_theta",
                    "units": "kg/m^3",
                },
            },
        )

        global_attributes = {
            "title": "Isopycnal Depth from Oceanic Pathways CROCO Model Output",
            "summary": (
                "This dataset contains the depths of the sigma_theta isopycnal surface "
                "referenced to the surface, i.e., sigma_0. "
                "Depths are computed using fields from the Oceanic Pathways model, "
                "a 1/36° regional CROCO model of the California Current System."
            ),
            "created_command": f"uv run {Path(__file__).name} isopycnal {self.target_sigma_0} "
            f"--save-dir {self.save_dir} --load-dir {self.load_dir}",
            "date_created": f"{datetime.now(UTC).isoformat()}",
            "history": {"Created": f"{datetime.now(UTC).isoformat()}"},
            "geospatial_lat_min": float(self.model["lat_rho"].min()),
            "geospatial_lat_max": float(self.model["lat_rho"].max()),
            "geospatial_lat_units": "degrees_north",
            "geospatial_lat_resolution": "1/36 degree",
            "geospatial_lon_min": float(self.model["lon_rho"].min()),
            "geospatial_lon_max": float(self.model["lon_rho"].max()),
            "geospatial_lon_units": "degrees_east",
            "geospatial_lon_resolution": "1/36 degree",
            **AUTHOR_ATTRIBUTES,
        }

        return variable_attributes, global_attributes

    def _process_slice(self, ds_slice: xr.Dataset) -> xr.DataArray:
        """Process a slice of the dataset to compute isopycnal depths.

        Args:
            ds_slice (xr.Dataset): Slice of the model dataset.

        Returns:
            xr.DataArray: DataArray containing the computed isopycnal depths for the slice.

        """
        ds_slice = compute_sigma_0(ds_slice, self.grid)
        # Construct a DataArray with the target sigma_0 value for each (eta_rho, xi_rho) point
        ds_slice["target_sigma_0"] = xr.full_like(ds_slice["sigma_0"].isel(s_rho=0), self.target_sigma_0)
        # Make sure dimensions are in the right order for apply_ufunc and chunking is correct
        ds_slice = ds_slice.transpose("time", "eta_rho", "xi_rho", "s_rho")
        ds_slice = ds_slice.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1, "s_rho": -1})

        # Now catch warnings for all-NaN slices
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"All-NaN (slice|axis) encountered")
            # Using vectorized=True and dask="parallelized"
            # This means that the function will be applied to each time independently in parallel
            # This is how the interpolate function is setup to work
            isopycnal_depth_slice: xr.DataArray = xr.apply_ufunc(
                self._interpolate_to_density,
                self.grid["z_rho"].transpose("eta_rho", "xi_rho", "s_rho"),  # Make sure dims are in correct order
                ds_slice["sigma_0"],
                ds_slice["target_sigma_0"],
                input_core_dims=[
                    ["eta_rho", "xi_rho", "s_rho"],
                    ["eta_rho", "xi_rho", "s_rho"],
                    ["eta_rho", "xi_rho"],
                ],
                output_core_dims=[["eta_rho", "xi_rho"]],
                output_dtypes=[ds_slice["sigma_0"].dtype],
                vectorize=True,
                dask="parallelized",
            )

        isopycnal_depth_slice = isopycnal_depth_slice.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1})
        isopycnal_depth_slice = isopycnal_depth_slice.rename("depth")

        return isopycnal_depth_slice

    def compute(self) -> None:
        """Compute the depth of a specified isopycnal surface (e.g., sigma_0 = 25.8 kg/m^3) from CROCO model output.

        Takes about 20 hours to run on a home desktop and saves ~20 GB of data
        when used on the Oceanic Pathways model output.

        """
        self.save_dir.mkdir(exist_ok=True)
        self.slices_dir.mkdir(exist_ok=True)

        with setup_client(self.n_workers, self.threads_per_worker, self.memory_limit) as client:
            typer.echo(f"Loading model files from {self.load_dir}...", nl=False)
            self.grid = self.open_grid_with_zdepths()
            self.model = self.open_model_fields()
            typer.echo("done.")

            # Find the number of time slices needed
            dataset_slices = self._slice_dataset(self.model)

            typer.echo("Computing and saving isopycnal depth slices...", nl=False)

            # Process each chunk of times separately
            for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing isopycnal depth slices")):
                isopycnal_depth_slice = self._process_slice(ds_slice)
                self._save_slice(isopycnal_depth_slice, i)

            # Setup final dataset with attributes
            typer.echo("Concatenating, adding attributes, and saving dataset...", nl=False)
            ds = self._concatenate_slices()
            ds = ds.expand_dims(sigma_theta=[self.target_sigma_0])
            var_attrs, global_attrs = self._generate_attributes()
            ds = self._assign_attributes(ds, var_attrs, global_attrs)
            self._save_dataset(ds)
            typer.echo("done.")

            typer.echo("All processing complete.")

            client.close()


class MixedLayerDepth(BaseInterpolation):
    """Class to compute mixed layer depths from CROCO model output."""

    def __init__(self, delta_sigma_0: float, save_dir: Path | str, load_dir: Path | str) -> None:
        """Initialize the MixedLayerDepth class.

        Args:
            delta_sigma_0 (float): The sigma_0 threshold for mixed layer depth calculation.
            save_dir (Path | str): The directory to save the computed mixed layer depths.
            load_dir (Path | str): The directory to load the model fields from.

        """
        self.delta_sigma_0 = delta_sigma_0
        super().__init__(save_dir, load_dir, time_slice_size=12 * 12)
        self.save_path = self.save_dir / f"mixed_layer_depth_delta_sigma_{self.delta_sigma_0}.zarr"
        self.slices_dir = self.save_path.with_name("mixed_layer_depth_slices")
        self.monthly_mean_path = self.save_dir / f"monthly_mean_{self.save_path.name}"

    def _generate_attributes(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate variable and global attributes for the isopycnal depth dataset.

        Returns:
            dict[str, Any]: A dictionary containing variable and global attributes.

        """
        variable_attributes = {str(var): self.model[var].attrs for var in self.model.sizes}

        variable_attributes.update(
            {
                "depth": {
                    "long_name": "thickness of the mixed layer with a "
                    f"threshold of sigma_0={self.delta_sigma_0} kg/m^3",
                    "standard_name": "ocean_mixed_layer_thickness_defined_by_sigma_theta",
                    "units": "m",
                },
                "threshold_sigma_theta": {
                    "long_name": "mixed layer threshold potential density referenced to the surface",
                    "standard_name": "sea_water_sigma_theta",
                    "units": "kg/m^3",
                },
            },
        )

        global_attributes = {
            "title": "Mixed Layer Depth from Oceanic Pathways CROCO Model Output",
            "summary": (
                "This dataset contains the thicknesses of the mixed layer defined by a sigma_theta threshold "
                "referenced to the surface, i.e., sigma_0. "
                "Depths are computed using fields from the Oceanic Pathways model, "
                "a 1/36° regional CROCO model of the California Current System."
            ),
            "created_command": f"uv run {Path(__file__).name} mld {self.delta_sigma_0} "
            f"--save-dir {self.save_dir} --load-dir {self.load_dir}",
            "date_created": f"{datetime.now(UTC).isoformat()}",
            "history": {"Created": f"{datetime.now(UTC).isoformat()}"},
            "geospatial_lat_min": float(self.model["lat_rho"].min()),
            "geospatial_lat_max": float(self.model["lat_rho"].max()),
            "geospatial_lat_units": "degrees_north",
            "geospatial_lat_resolution": "1/36 degree",
            "geospatial_lon_min": float(self.model["lon_rho"].min()),
            "geospatial_lon_max": float(self.model["lon_rho"].max()),
            "geospatial_lon_units": "degrees_east",
            "geospatial_lon_resolution": "1/36 degree",
            **AUTHOR_ATTRIBUTES,
        }

        return variable_attributes, global_attributes

    def _process_slice(self, ds_slice: xr.Dataset) -> xr.DataArray:
        """Process a slice of the dataset to compute isopycnal depths.

        Args:
            ds_slice (xr.Dataset): Slice of the model dataset.

        Returns:
            xr.DataArray: DataArray containing the computed isopycnal depths for the slice.

        """
        ds_slice = compute_sigma_0(ds_slice, self.grid)
        # grid is from bottom to top so select last index
        ds_slice["threshold_sigma_0"] = ds_slice["sigma_0"].isel(s_rho=-1) + self.delta_sigma_0
        # Make sure dimensions are in the right order for apply_ufunc and chunking is correct
        ds_slice = ds_slice.transpose("time", "eta_rho", "xi_rho", "s_rho")
        ds_slice = ds_slice.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1, "s_rho": -1})

        # Now catch warnings for all-NaN slices
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"All-NaN (slice|axis) encountered")
            # Using vectorized=True and dask="parallelized"
            # This means that the function will be applied to each time independently in parallel
            # This is how the interpolate function is setup to work
            mixed_layer_depth_slice: xr.DataArray = xr.apply_ufunc(
                self._interpolate_to_density,
                self.grid["z_rho"].transpose("eta_rho", "xi_rho", "s_rho"),  # Make sure dims are in correct order
                ds_slice["sigma_0"],
                ds_slice["threshold_sigma_0"],
                input_core_dims=[
                    ["eta_rho", "xi_rho", "s_rho"],
                    ["eta_rho", "xi_rho", "s_rho"],
                    ["eta_rho", "xi_rho"],
                ],
                output_core_dims=[["eta_rho", "xi_rho"]],
                output_dtypes=[ds_slice["sigma_0"].dtype],
                vectorize=True,
                dask="parallelized",
            )

        mixed_layer_depth_slice = mixed_layer_depth_slice.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1})
        mixed_layer_depth_slice = mixed_layer_depth_slice.rename("depth")

        return mixed_layer_depth_slice

    def compute(self) -> None:
        """Compute and save the mixed layer depth from CROCO model output using the threshold method."""
        self.save_dir.mkdir(exist_ok=True)
        self.slices_dir.mkdir(exist_ok=True)

        with setup_client(self.n_workers, self.threads_per_worker, self.memory_limit) as client:
            typer.echo(f"Loading model files from {self.load_dir}...", nl=False)
            self.grid = self.open_grid_with_zdepths()
            self.model = self.open_model_fields()
            typer.echo("done.")

            # Slice the dataset into manageable time chunks
            dataset_slices = self._slice_dataset(self.model)

            typer.echo("Computing and saving mixed layer depth slices...", nl=False)

            # Process each chunk of times separately
            for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing mixed layer depth slices")):
                mixed_layer_depth_slice = self._process_slice(ds_slice)
                self._save_slice(mixed_layer_depth_slice, i)

            # Setup final dataset with attributes
            typer.echo("Concatenating, adding attributes, and saving dataset...", nl=False)
            ds = self._concatenate_slices()
            ds = ds.expand_dims(threshold_sigma_theta=[self.delta_sigma_0])
            var_attrs, global_attrs = self._generate_attributes()
            ds = self._assign_attributes(ds, var_attrs, global_attrs)
            self._save_dataset(ds)
            typer.echo("done.")

            typer.echo("All processing complete.")

            client.close()


class DensityAtMixedLayerDepth(BaseInterpolation):
    """Class to compute density at mixed layer depths from CROCO model output."""

    def __init__(self, mixed_layer_depth: MixedLayerDepth) -> None:
        """Initialize the DensityAtMixedLayerDepth class.

        Args:
            mixed_layer_depth (MixedLayerDepth): An instance of the MixedLayerDepth class.

        Raises:
            FileNotFoundError: If the mixed layer depth file does not exist.

        """
        self.mixed_layer_depth_cls = mixed_layer_depth
        super().__init__(mixed_layer_depth.save_dir, mixed_layer_depth.load_dir, time_slice_size=12 * 12)
        self.save_path = self.save_dir / f"density_at_mld_delta_sigma_{mixed_layer_depth.delta_sigma_0}.zarr"
        self.slices_dir = self.save_path.with_name("density_at_mld_slices")
        self.monthly_mean_path = self.save_dir / f"monthly_mean_{self.save_path.name}"

        if not self.mixed_layer_depth_cls.save_path.exists():
            msg = f"Mixed layer depth file {self.mixed_layer_depth_cls.save_path} does not exist."
            msg += "Please compute it with `MixedLayerDepth.compute()`."
            raise FileNotFoundError(msg)

        self.mixed_layer_depth = self.mixed_layer_depth_cls.open()["depth"].squeeze()

    def _generate_attributes(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate variable and global attributes for the isopycnal depth dataset.

        Returns:
            dict[str, Any]: A dictionary containing variable and global attributes.

        """
        variable_attributes = {str(var): self.model[var].attrs for var in self.model.sizes}

        variable_attributes.update(
            {
                "sigma_theta": {
                    "long_name": "sigma_theta at the mixed layer depth defined with a "
                    f"threshold of sigma_0={self.mixed_layer_depth_cls.delta_sigma_0} kg/m^3",
                    "standard_name": "sea_water_sigma_theta",
                    "units": "kg/m^3",
                },
                "threshold_sigma_theta": {
                    "long_name": "mixed layer threshold potential density referenced to the surface",
                    "standard_name": "sea_water_sigma_theta",
                    "units": "kg/m^3",
                },
            },
        )

        global_attributes = {
            "title": "Density at mixed layer depth from Oceanic Pathways CROCO Model Output",
            "summary": (
                "This dataset contains the density at the mixed layer depth defined by a sigma_theta threshold "
                "referenced to the surface, i.e., sigma_0. "
                "Depths are computed using fields from the Oceanic Pathways model, "
                "a 1/36° regional CROCO model of the California Current System."
            ),
            "mixed_layer_depth_file": str(self.mixed_layer_depth_cls.save_path),
            "created_command": f"uv run {Path(__file__).name} mld_density {self.mixed_layer_depth_cls.delta_sigma_0} "
            f"--save-dir {self.save_dir} --load-dir {self.load_dir}",
            "date_created": f"{datetime.now(UTC).isoformat()}",
            "history": {"Created": f"{datetime.now(UTC).isoformat()}"},
            "geospatial_lat_min": float(self.model["lat_rho"].min()),
            "geospatial_lat_max": float(self.model["lat_rho"].max()),
            "geospatial_lat_units": "degrees_north",
            "geospatial_lat_resolution": "1/36 degree",
            "geospatial_lon_min": float(self.model["lon_rho"].min()),
            "geospatial_lon_max": float(self.model["lon_rho"].max()),
            "geospatial_lon_units": "degrees_east",
            "geospatial_lon_resolution": "1/36 degree",
            **AUTHOR_ATTRIBUTES,
        }

        return variable_attributes, global_attributes

    def _process_slice(self, ds_slice: xr.Dataset) -> xr.DataArray:
        """Process a slice of the dataset to compute density at mixed layer depths.

        Args:
            ds_slice (xr.Dataset): Slice of the model dataset.

        Returns:
            xr.DataArray: DataArray containing the computed densities at mixed layer depths for the slice.

        """
        ds_slice = compute_sigma_0(ds_slice, self.grid)
        # Make sure dimensions are in the correct order for interpolation
        ds_slice = ds_slice.transpose("time", "eta_rho", "xi_rho", "s_rho")

        # Now catch warnings for all-NaN slices
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"All-NaN (slice|axis) encountered")
            # Using vectorized=True and dask="parallelized"
            # This means that the function will be applied to each time independently in parallel
            # This is how the interpolate function is setup to work
            density_at_mld_slice: xr.DataArray = xr.apply_ufunc(
                self._interpolate_to_depth,
                self.grid["z_rho"].transpose("eta_rho", "xi_rho", "s_rho"),  # same as above comment
                ds_slice["sigma_0"],
                ds_slice["mld"],
                input_core_dims=[
                    ["eta_rho", "xi_rho", "s_rho"],
                    ["eta_rho", "xi_rho", "s_rho"],
                    ["eta_rho", "xi_rho"],
                ],
                output_core_dims=[["eta_rho", "xi_rho"]],
                output_dtypes=[ds_slice["sigma_0"].dtype],
                vectorize=True,
                dask="parallelized",
            )

        # Make sure output is chunked appropriately
        density_at_mld_slice = density_at_mld_slice.chunk({"time": 1, "eta_rho": -1, "xi_rho": -1})
        density_at_mld_slice = density_at_mld_slice.rename("density")

        return density_at_mld_slice

    def compute(self) -> None:
        """Compute and save the mixed layer depth from CROCO model output using the threshold method."""
        self.save_dir.mkdir(exist_ok=True)
        self.slices_dir.mkdir(exist_ok=True)

        with setup_client(self.n_workers, self.threads_per_worker, self.memory_limit) as client:
            typer.echo(f"Loading model files from {self.load_dir}...", nl=False)
            self.grid = self.open_grid_with_zdepths()
            self.model = self.open_model_fields()
            typer.echo("done.")

            # Add mixed layer depth to the model dataset for interpolation
            self.model["mld"] = self.mixed_layer_depth

            # Slice the dataset into manageable time chunks
            dataset_slices = self._slice_dataset(self.model)

            typer.echo("Computing and saving mixed layer depth slices...", nl=False)

            # Process each chunk of times separately
            for i, ds_slice in enumerate(tqdm(dataset_slices, desc="Computing mixed layer depth slices")):
                mixed_layer_depth_slice = self._process_slice(ds_slice)
                self._save_slice(mixed_layer_depth_slice, i)

            # Setup final dataset with attributes
            typer.echo("Concatenating, adding attributes, and saving dataset...", nl=False)
            ds = self._concatenate_slices()
            ds = ds.expand_dims(threshold_sigma_theta=[self.mixed_layer_depth_cls.delta_sigma_0])
            var_attrs, global_attrs = self._generate_attributes()
            ds = self._assign_attributes(ds, var_attrs, global_attrs)
            self._save_dataset(ds)
            typer.echo("done.")

            typer.echo("All processing complete.")

            client.close()


# Setup Typer CLI app
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
        typer.echo(f"Computing isopycnal depth for sigma_0 = {value}")
        iso.compute()

    elif mode == "mld":
        mld = MixedLayerDepth(delta_sigma_0=float(value), save_dir=save_dir, load_dir=load_dir)
        typer.echo(f"Computing mixed layer depth with delta_sigma = {value}")
        mld.compute()

    elif mode == "mld_density":
        # Ensure MLD is available; compute if not present
        mld = MixedLayerDepth(delta_sigma_0=float(value), save_dir=save_dir, load_dir=load_dir)
        if not mld.save_path.exists():
            typer.echo("Mixed layer depth file not found. ", nl=False)
            typer.echo("Enter 'y' to compute it now, 'n' to exit, or provide the path to an existing file: ", nl=False)
            user_input = input()
            if user_input.lower() == "y":
                mld.compute()
            elif user_input.lower() == "n":
                typer.echo("Exiting.")
                raise typer.Exit(code=1)
            elif Path(user_input).exists():
                mld = MixedLayerDepth(
                    delta_sigma_0=float(value),
                    save_dir=Path(user_input).parent,
                    load_dir=load_dir,
                )
            else:
                typer.echo("Cannot compute density at MLD without mixed layer depth. Exiting.")
                raise typer.Exit(code=1)
        density = DensityAtMixedLayerDepth(mld)
        typer.echo(f"Computing density at MLD for delta_sigma = {value}")
        density.compute()


if __name__ == "__main__":
    # Run the Typer CLI app
    app()
