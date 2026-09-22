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
"""
import argparse
import os
import shutil
import time

import pandas as pd
import xarray as xr

import config
import regional_mom6 as rmom6

# Confirms which regional_mom6 copy got loaded (dev checkout vs conda env).
print(f"Using regional_mom6 from: {rmom6.__file__}")


def get_day_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-index", type=int, default=None)
    args = parser.parse_args()
    if args.day_index is not None:
        return args.day_index
    env_idx = os.environ.get("PBS_ARRAY_INDEX")
    if env_idx is not None:
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
    day_work_dir.mkdir(parents=True, exist_ok=True)

    # Private per-day mom_run_dir too. experiment.__init__() symlinks
    # mom_run_dir/"inputdir" -> mom_input_dir only if it doesn't already
    # exist -- sharing one mom_run_dir across days let the first day's
    # symlink go dangling once its folder was deleted, which then crashed
    # the next day to run with FileExistsError. A private, always-fresh
    # mom_run_dir per day removes the shared path entirely.
    day_run_dir = config.daily_stage_dir / f"_rundir_{day_str}"
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
    uv_day_ds = expt.ic_vels.expand_dims(time=[t])
    tracer_day_ds.to_netcdf(tracer_out_path)
    uv_day_ds.to_netcdf(uv_out_path)
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
