"""Primary script to run all processing steps for isopycnal depth calculations.

Runs the following steps in order:
 1. (Optional) Compute and save z-level depths and pressures to grid dataset using `compute_z_levels.py`.
 2. Compute and save isopycnal depths to a zarr store using `compute_isopycnal_depth.py`.
 3. Compute and save monthly mean isopycnal depths to a zarr store using `calculate_monthly_mean_isopycnal_depth.py`.
"""

from calculate_monthly_mean_isopycnal_depth import save_monthly_mean_isopycnal_depth
from compute_isopycnal_depth import compute_isopycnal_depth
from compute_z_levels import compute_depth_and_pressure

if __name__ == "__main__":
    parent_path = "D:/avg/"

    # Step 1: (Optional) Compute and save z-level depths and pressures to grid dataset
    compute_depth_and_pressure(parent_path)

    # Step 2: Compute and save isopycnal depths to a zarr store
    target_sigma_0 = 25.8
    time_slice_size = 100
    compute_isopycnal_depth(parent_path, target_sigma_0, time_slice_size)

    # Step 3: Compute and save monthly mean isopycnal depths to a zarr store
    save_monthly_mean_isopycnal_depth()
