"""Compute z-depths and pressure levels for a CROCO model and save to a new grid file.

Intended to be run before `compute_isopycnal_depth.py` to ensure z-depths and pressures are available.
"""

from enum import Enum, auto
from pathlib import Path

import gsw
import numpy as np
import xarray as xr

from utils import open_grid, open_model_fields


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


if __name__ == "__main__":
    parent_path = "D:/avg/"

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

    ds_grid.to_netcdf(Path(parent_path) / "croco_grd_with_z.nc")
