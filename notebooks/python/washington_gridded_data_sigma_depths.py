# ---
# jupyter:
#   jupytext:
#     formats: ipynb,python//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: analysis
#     language: python
#     name: python3
# ---

# %%
from functools import partial
from pathlib import Path
from typing import cast

import gsw
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import brentq as find_root

# %%
SIGMA_LAYER = 25.8  # kg/m^3; potential density of isopycnal surface to find the depth of
MLD_THRESHOLD = 0.1  # kg/m^3; threshold for mixed layer depth calculation

# %%
DATA_DIRECTORY = Path("../data")
CE09_OSPM_FILE = DATA_DIRECTORY / "ce09ospm_gridded_profiles.nc"

ce09 = xr.open_dataset(CE09_OSPM_FILE)
ce09 = ce09.squeeze().load()
ce09 = ce09.swap_dims({"pressure": "depth"})
# backfill surface data for identifying surface sigma_0, but limit to 5 m depth (10 data points at 0.5 m intervals)
ce09_bfill = ce09.bfill("depth", limit=10)
ce09["threshold_sigma_0"] = ce09_bfill["potential_density"].isel(depth=0) + MLD_THRESHOLD
ce09["salinity_absolute"] = gsw.SA_from_SP(
    ce09["salinity"],
    ce09["pressure"],
    ce09["longitude"],
    ce09["latitude"],
)
ce09["conservative_temperature"] = gsw.CT_from_t(
    ce09["salinity_absolute"],
    ce09["temperature"],
    ce09["pressure"],
)
ce09["spice"] = gsw.spiciness0(ce09["salinity_absolute"], ce09["conservative_temperature"])


# %%
def calculate_mld(depth: np.ndarray, z: np.ndarray, sigma_0: np.ndarray, surface_sigma_0: float) -> float:
    """Calculate mixed layer depth for a single time step.

    Args:
        depth (np.ndarray): Array of depths at which to calculate the mixed layer depth.
        z (np.ndarray): Array of depths corresponding to sigma_0.
        sigma_0 (np.ndarray): Array of sigma_0 values.
        surface_sigma_0 (float): Surface sigma_0 value.

    Returns:
        float: Mixed layer depth.

    """
    mld = np.interp(depth, z, sigma_0, left=np.nan, right=np.nan) - surface_sigma_0
    return float(mld)


mld = np.full(ce09["time"].size, np.nan)
sigma_layer_depth = np.full(ce09["time"].size, np.nan)

for i, (sigma_0, surface_sigma_0) in enumerate(
    zip(
        ce09["potential_density"].isel(depth=slice(None)).to_numpy().T,
        ce09["threshold_sigma_0"].to_numpy(),
        strict=True,
    ),
):
    z = ce09["depth"].to_numpy()[~np.isnan(sigma_0)]
    sigma_0_filtered = sigma_0[~np.isnan(sigma_0)]
    try:
        mld[i] = find_root(
            partial(
                calculate_mld,
                z=z,
                sigma_0=sigma_0_filtered,
                surface_sigma_0=surface_sigma_0,
            ),
            np.min(z),
            np.max(z),
        )
    except ValueError:
        mld[i] = np.nan
    try:
        sigma_layer_depth[i] = find_root(
            partial(
                calculate_mld,
                z=z,
                sigma_0=sigma_0_filtered,
                surface_sigma_0=SIGMA_LAYER,
            ),
            np.min(z),
            np.max(z),
        )
    except ValueError:
        sigma_layer_depth[i] = np.nan

sigma_at_mld = ce09["potential_density"].interp(depth=xr.DataArray(mld, dims="time"))

# %%
ce09

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
ax1: plt.Axes = axs[0]
ax2: plt.Axes = axs[1]

ax1.plot(ce09["time"], -mld, label="Mixed Layer Depth")
ax1.plot(ce09["time"], -sigma_layer_depth, label=f"Depth of {SIGMA_LAYER}$\\sigma_0$")
ax1.set_ylabel("z (m)")
ax1.legend()

ax2.plot(ce09["time"], sigma_layer_depth - mld)
ax2.set_ylabel(f"{SIGMA_LAYER}$\\sigma_0$ depth - MLD (m)")
ax2.annotate(
    "$\\uparrow$ MLD shallower",
    xy=(0.005, 0.08),
    xycoords="axes fraction",
    ha="left",
)
ax2.axhline(0, color="k", linestyle="--")

plt.savefig("../misc/wa_profiler_mld_sigma_layer_depth.png", dpi=300)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 4), layout="constrained")

ax.plot(ce09["time"], sigma_at_mld)
ax.set_ylabel("Potential density $\\sigma_0$ at MLD (kg/m$^3$)")
ax.axhline(SIGMA_LAYER, color="k", linestyle="--")
plt.savefig("../misc/wa_profiler_sigma_at_mld.png", dpi=300)


# %% [markdown]
# ## Time series along 25.8 and 26.0

# %%
SIGMA_LAYERS = [25.8, 26.0]


# %%
def calculate_mld(depth: np.ndarray, z: np.ndarray, sigma_0: np.ndarray, surface_sigma_0: float) -> float:
    """Calculate mixed layer depth for a single time step.

    Args:
        depth (np.ndarray): Array of depths at which to calculate the mixed layer depth.
        z (np.ndarray): Array of depths corresponding to sigma_0.
        sigma_0 (np.ndarray): Array of sigma_0 values.
        surface_sigma_0 (float): Surface sigma_0 value.

    Returns:
        float: Mixed layer depth.

    """
    mld = np.interp(depth, z, sigma_0, left=np.nan, right=np.nan) - surface_sigma_0
    return float(mld)


sigma_layers_depth_array = np.full((len(SIGMA_LAYERS), ce09["time"].size), np.nan)

for i, sigma_0 in enumerate(
    ce09["potential_density"].isel(depth=slice(None)).to_numpy().T,
):
    z = ce09["depth"].to_numpy()[~np.isnan(sigma_0)]
    sigma_0_filtered = sigma_0[~np.isnan(sigma_0)]
    for j, sigma_layer in enumerate(SIGMA_LAYERS):
        try:
            sigma_layers_depth_array[j, i] = find_root(
                partial(
                    calculate_mld,
                    z=z,
                    sigma_0=sigma_0_filtered,
                    surface_sigma_0=sigma_layer,
                ),
                np.min(z),
                np.max(z),
            )
        except ValueError:
            sigma_layers_depth_array[j, i] = np.nan

sigma_layers_depth_da = xr.DataArray(
    sigma_layers_depth_array,
    dims=["sigma_layer", "time"],
    coords={"sigma_layer": SIGMA_LAYERS, "time": ce09["time"]},
)


# %%
oxy_interp = ce09["dissolved_oxygen"].interp(depth=sigma_layers_depth_da)
temp_interp = ce09["conservative_temperature"].interp(depth=sigma_layers_depth_da)
sal_interp = ce09["salinity"].interp(depth=sigma_layers_depth_da)
spice_interp = ce09["spice"].interp(depth=sigma_layers_depth_da)

# %%
fig, axs = plt.subplots(3, 1, figsize=(8, 6), layout="constrained", sharex=True)
ax0 = cast("plt.Axes", axs[0])
ax1 = cast("plt.Axes", axs[1])
ax2 = cast("plt.Axes", axs[2])

ax0.plot(ce09["time"], -sigma_layers_depth_da.sel(sigma_layer=25.8), label="25.8 $\\mathsf{kg/m^3}$", c="#4477AA")
ax0.plot(ce09["time"], -sigma_layers_depth_da.sel(sigma_layer=26.0), label="26.0 $\\mathsf{kg/m^3}$", c="#CC6677")
ax1.plot(ce09["time"], spice_interp.sel(sigma_layer=25.8), c="#4477AA")
ax1.plot(ce09["time"], spice_interp.sel(sigma_layer=26.0), c="#CC6677")
ax2.plot(ce09["time"], oxy_interp.sel(sigma_layer=25.8), c="#4477AA")
ax2.plot(ce09["time"], oxy_interp.sel(sigma_layer=26.0), c="#CC6677")

ax0.set_ylabel("z (m)")
ax1.set_ylabel("Spice ($\\mathsf{kg/m^3}$)")
ax2.set_ylabel("Oxygen ($\\mathsf{\\mu mol/kg}$)")

ax0.legend(framealpha=1, ncols=2, borderpad=0.2)
ax1.axhline(0, color="k", linestyle="-")

fig.suptitle("Washington offshore profiler water properties on isopycnal surfaces")

plt.savefig("../misc/wa_offshore_profiler_isopycnal_properties_timeseries.png", dpi=300)
