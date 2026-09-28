#!/usr/bin/env python3
"""
Process a single day's initial condition and write it to the shared staging
directory as its own uniquely-named file (tracers + u/v; eta is dropped --
not needed, but setup_initial_condition() writes it as a side effect anyway).

Runs as one task of a PBS job array, so days process in parallel instead of
in a long serial loop. Run 00_build_grid.py once first.

Usage:
    python3 02_process_one_day.py --day-index 0
    (or omit --day-index to read it from $PBS_ARRAY_INDEX, set automatically
    by PBS when this runs as one task of a job array)

u and v are written on the h (tracer) points -- (zl, ny, nx), same as temp/salt
-- so 06_uv_to_hpoints.py is no longer needed.
"""
import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr

import config
import regional_mom6 as rmom6
from regional_mom6 import regridding as rgd
from regional_mom6.utils import rotate, try_pint_convert

# Confirms which regional_mom6 copy got loaded (dev checkout vs conda env).
print(f"Using regional_mom6 from: {rmom6.__file__}")


def uv_on_h_points(expt, raw_path, varnames, regridding_method=None):
    """Regrid A-grid u/v straight onto MOM6 h points (the tracer points).

    Follows the same steps as expt.setup_initial_condition() -- unit
    conversion, NaN filling, horizontal regrid, rotation to the model grid,
    vertical interpolation, missing-data fill -- but samples the h points
    instead of the staggered u/v faces. Returns a Dataset with u, v on
    (zl, ny, nx) and the same xh/yh/nx/ny coordinates as expt.ic_tracers.
    """
    if regridding_method is None:
        regridding_method = expt.regridding_method
    vm = rgd.apply_arakawa_grid_mapping(var_mapping=varnames, arakawa_grid="A")
    x, y, z = vm["tracer_x_coord"], vm["tracer_y_coord"], vm["depth_coord"]
    if isinstance(z, list):
        z = z[0]
    tname = vm["time_var_name"]

    ds = xr.open_dataset(raw_path)
    if tname in ds.dims:
        ds = ds.isel({tname: 0})
    ds = ds.drop_vars(tname, errors="ignore")
    ds[z] = try_pint_convert(ds[z], "m", z)

    uv = {}
    for name in ("u", "v"):
        da = try_pint_convert(ds[vm[f"{name}_var_name"]], "m/s", name)
        # Same land-NaN filling the library applies before regridding.
        da = (
            da.interpolate_na(x, method="linear")
            .ffill(x).bfill(x).ffill(y).bfill(y).ffill(z)
        )
        uv[name] = da.rename({vm["u_lon_coord"]: "lon", vm["u_lat_coord"]: "lat"})

    # h points of the supergrid (identical to the tgrid used for tracers).
    hgrid = expt.hgrid
    hgrid["lon"] = hgrid["x"]
    hgrid["lat"] = hgrid["y"]
    tpts = rgd.get_hgrid_arakawa_c_points(hgrid, "t")
    tgrid = tpts.rename(
        {"tlon": "lon", "tlat": "lat", "nxp": "nx", "nyp": "ny"}
    ).set_coords(["lat", "lon"])

    regridder = rgd.create_regridder(
        uv["u"], tgrid, locstream_out=False, method=regridding_method
    )
    u_h = regridder(uv["u"])
    v_h = regridder(uv["v"])

    # Rotate east/north velocities onto the model grid, using the grid angle
    # at the h points (the library uses the angle at the u/v points).
    angle_h = np.radians(
        hgrid.angle_dx.values[tpts.t_points_y.values][:, tpts.t_points_x.values]
    )
    u_h, v_h = rotate(u_h, v_h, radian_angle=angle_h)

    out = (
        xr.Dataset({"u": u_h, "v": v_h})
        .rename({"lon": "xh", "lat": "yh", z: "zl"})
        .transpose("zl", "ny", "nx", ...)
    )
    out = out.assign_coords(
        nx=np.arange(out.sizes["nx"]).astype(float),
        ny=np.arange(out.sizes["ny"]).astype(float),
    )
    out = out.interp({"zl": expt.vgrid.zl.values}, kwargs={"fill_value": "extrapolate"})
    out = rgd.fill_missing_data(out, "all")
    out["u"].attrs = ds[vm["u_var_name"]].attrs
    out["v"].attrs = ds[vm["v_var_name"]].attrs
    ds.close()
    return out


def get_day_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-index", type=int, default=None)
    args = parser.parse_args()
    if args.day_index is not None:
        return args.day_index
    # PBS_ARRAY_INDEX inside an array job; DAY_INDEX for the single-day
    # (non-array) jobs submit_all.sh uses when only one day is left in a run.
    for var in ("PBS_ARRAY_INDEX", "DAY_INDEX"):
        env_idx = os.environ.get(var)
        if env_idx:
            return int(env_idx)
    raise SystemExit(
        "No day index given -- pass --day-index N, or run this under a PBS "
        "job array that sets $PBS_ARRAY_INDEX."
    )


def main():
    day_index = get_day_index()
    t_start = time.time()
    config.ensure_dirs()

    with xr.open_dataset(config.ic_source_path) as ic_full:
        time_values = ic_full[config.ocean_varnames["time"]].values

    if day_index < 0 or day_index >= len(time_values):
        raise SystemExit(
            f"day_index {day_index} out of range -- source file has "
            f"{len(time_values)} timesteps (valid range 0..{len(time_values) - 1})"
        )

    t = time_values[day_index]
    day_str = pd.Timestamp(t).strftime("%Y%m%d")
    print(f"=== Day {day_index}: {day_str} (pid {os.getpid()}) ===")

    tracer_out_path = config.daily_stage_dir / f"ic_tracers_{day_str}.nc"
    uv_out_path = config.daily_stage_dir / f"ic_uv_{day_str}.nc"
    if tracer_out_path.exists() and uv_out_path.exists():
        print(f"Day {day_str} already done, skipping.")
        return

    # Private per-day scratch dir -- setup_initial_condition() always writes
    # init_eta.nc/init_tracers.nc/init_vel.nc under fixed names, so sharing
    # one dir across parallel tasks would make them race to write the same
    # files.
    day_work_dir = config.daily_stage_dir / f"_work_{day_str}"
    # Start clean: a previous attempt at this day may have been killed
    # half-way and left files behind.
    shutil.rmtree(day_work_dir, ignore_errors=True)
    day_work_dir.mkdir(parents=True, exist_ok=True)

    # Private per-day mom_run_dir too. experiment.__init__() symlinks
    # mom_run_dir/"inputdir" -> mom_input_dir only if it doesn't already
    # exist -- sharing one mom_run_dir across days let the first day's
    # symlink go dangling once its folder was deleted, which then crashed
    # the next day to run with FileExistsError. A private, always-fresh
    # mom_run_dir per day removes the shared path entirely.
    day_run_dir = config.daily_stage_dir / f"_rundir_{day_str}"
    shutil.rmtree(day_run_dir, ignore_errors=True)
    day_run_dir.mkdir(parents=True, exist_ok=True)
    day_experiment_kwargs = {**config.EXPERIMENT_KWARGS, "mom_run_dir": day_run_dir}

    # Reuse the pre-built horizontal grid (see 00_build_grid.py) instead of
    # regenerating it.
    src = config.input_dir / "hgrid.nc"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found -- run 00_build_grid.py first.")
    shutil.copy2(src, day_work_dir / "hgrid.nc")

    # Vertical grid is regenerated per day rather than reused: the vgrid
    # property is asymmetric -- vgrid_type="from_file" looks for vgrid.nc,
    # but writing that same property always saves to vcoord.nc. So there's
    # no vgrid.nc this pipeline ever produces for "from_file" to reload.
    # Regenerating is deterministic (same EXPERIMENT_KWARGS every time) and
    # fast (pure NumPy), so there's no real cost to doing it per day.
    expt = rmom6.experiment(
        mom_input_dir=day_work_dir,
        hgrid_type="from_file",
        vgrid_type="hyperbolic_tangent",
        **day_experiment_kwargs,
    )

    # 1. Slice out just this one timestep and save it as its own small file.
    temp_ic_path = day_work_dir / "ic_unprocessed_single_day.nc"
    with xr.open_dataset(config.ic_source_path) as ic_full:
        ic_single_day = ic_full.sel({config.ocean_varnames["time"]: [t]})
        ic_single_day.to_netcdf(temp_ic_path)

    # 2. Run the existing regional-mom6 processing on this single-day file.
    expt.setup_initial_condition(
        temp_ic_path,
        config.ocean_varnames,
        arakawa_grid="A",
    )

    # 3. Tag with the day's timestamp and write to the shared staging dir
    # under a unique, day-stamped name.
    tracer_day_ds = expt.ic_tracers.expand_dims(time=[t])
    # u/v on h points (not expt.ic_vels, which is on the staggered u/v faces).
    uv_day_ds = uv_on_h_points(expt, temp_ic_path, config.ocean_varnames).expand_dims(
        time=[t]
    )
    # Write to .tmp and rename only when complete, so a job killed mid-write
    # (walltime, node failure) never leaves a truncated file that the
    # "already done" check above -- and 01_list_missing_days.py -- would
    # mistake for a finished day.
    for ds_out, final in ((tracer_day_ds, tracer_out_path), (uv_day_ds, uv_out_path)):
        tmp = final.with_suffix(".nc.tmp")
        ds_out.to_netcdf(tmp)
        os.replace(tmp, final)
    tracer_day_ds.close()
    uv_day_ds.close()

    # Clean up this task's private scratch dirs.
    shutil.rmtree(day_work_dir, ignore_errors=True)
    shutil.rmtree(day_run_dir, ignore_errors=True)

    print(
        f"Day {day_str} done in {time.time() - t_start:.1f}s -> "
        f"{tracer_out_path.name}, {uv_out_path.name}"
    )


if __name__ == "__main__":
    main()
