#!/usr/bin/env python3
"""
Concatenate the per-day tracer/u-v files from 02_process_one_day.py into the
two final time-concatenated output files.

Run only after every array task in process_days.pbs has finished
successfully (submit_all.sh wires this up via a PBS job dependency). Pure
xarray concatenation -- no regional_mom6, no grid -- so it's cheap and safe
to re-run.

Each per-day tracer file is ~3.5GB and each uv file ~3.4GB; across ~370
days that's ~1.3TB + ~1.26TB (~2.5TB total) -- too big to load into memory,
so both files are read and written lazily, one day-chunk at a time via dask.
"""
import os
import sys

import pandas as pd
import xarray as xr

import config


def expected_day_strings():
    with xr.open_dataset(config.ic_source_path) as ic_full:
        time_values = ic_full[config.ocean_varnames["time"]].values
    return [pd.Timestamp(t).strftime("%Y%m%d") for t in time_values]


def main():
    day_strs = expected_day_strings()
    print(f"Expecting {len(day_strs)} days, from {day_strs[0]} to {day_strs[-1]}")

    tracer_files = [config.daily_stage_dir / f"ic_tracers_{d}.nc" for d in day_strs]
    uv_files = [config.daily_stage_dir / f"ic_uv_{d}.nc" for d in day_strs]

    missing = [f for f in tracer_files + uv_files if not f.exists()]
    if missing:
        print(
            f"ERROR: {len(missing)} expected per-day file(s) are missing. "
            "Some 02_process_one_day.py array tasks may not have finished "
            "(or may have failed). First few missing:"
        )
        for f in missing[:10]:
            print(f"  {f}")
        sys.exit(1)

    # Fail fast if the final outputs already exist, before doing any work.
    for out_path in (config.ic_tracers_output_path, config.ic_uv_output_path):
        if out_path.exists():
            raise FileExistsError(
                f"{out_path} already exists -- refusing to overwrite it. "
                "Move/rename/delete the existing file first if you want to "
                "regenerate it."
            )

    # Write to a .tmp path and rename into place only on success, so a job
    # killed partway through never leaves a corrupt file at the final path.
    tracer_tmp_path = config.ic_tracers_output_path.with_suffix(".nc.tmp")
    uv_tmp_path = config.ic_uv_output_path.with_suffix(".nc.tmp")

    print("Concatenating daily tracer files...")
    # chunks={"time": 1} keeps dask chunks aligned with the per-day files,
    # so to_netcdf() streams the write out day-by-day.
    with xr.open_mfdataset(
        tracer_files, combine="nested", concat_dim="time", chunks={"time": 1}
    ) as combined_tracers:
        config.ic_tracers_output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_tracers.to_netcdf(tracer_tmp_path)
    os.rename(tracer_tmp_path, config.ic_tracers_output_path)
    print(f"Saved concatenated tracer IC file to: {config.ic_tracers_output_path}")

    print("Concatenating daily u/v files...")
    with xr.open_mfdataset(
        uv_files, combine="nested", concat_dim="time", chunks={"time": 1}
    ) as combined_uv:
        config.ic_uv_output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_uv.to_netcdf(uv_tmp_path)
    os.rename(uv_tmp_path, config.ic_uv_output_path)
    print(f"Saved concatenated u/v IC file to: {config.ic_uv_output_path}")

    print("Cleaning up per-day files and the staging directory...")
    for f in tracer_files + uv_files:
        f.unlink()
    try:
        config.daily_stage_dir.rmdir()
        print(f"Removed staging directory: {config.daily_stage_dir}")
    except OSError:
        print(f"Note: {config.daily_stage_dir} was not empty, left in place for inspection.")

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
