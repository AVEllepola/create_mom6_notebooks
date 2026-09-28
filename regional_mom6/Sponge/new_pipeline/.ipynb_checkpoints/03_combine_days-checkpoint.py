#!/usr/bin/env python3
"""
Combine the per-day tracer/u-v files from 02_process_one_day.py into the two
final sponge/nudging files in ONE pass. Replaces:
  * 03_concatenate_days.py        (stack the days along time)
  * 04_make_time_unlimited.py     (time -> unlimited record dimension)
  * 05_fix_dimensions_like_gfdl.py (GFDL-style coordinates/attributes)
  * 06_uv_to_hpoints.py            (not needed: 02 now writes u/v on h points)

Output files (both tracers and u/v), GFDL nudging-file style:
  * u, v, temp, salt, ... on (time, zl, ny, nx)
  * time: UNLIMITED, float64, calendar "gregorian", cartesian_axis "T"
  * zl: cartesian_axis "Z", positive "down"
  * nx / ny: 1-D coordinates with cartesian_axis "X" / "Y"; real lon/lat
    (degrees_east / degrees_north) when the grid is separable (lon depends
    only on x, lat only on y), otherwise index values 1..N
  * one horizontal slab per HDF5 chunk, no compression, no _FillValue on
    coordinate variables

Data is streamed one day at a time with dask, so memory stays small even
though each output is ~1.3 TB. Each output is written to *.nc.tmp, checked,
then renamed into place, so a killed job never leaves a half-written file.

Run only after every array task of process_days.pbs has finished.
"""
import os
import sys
import time as _time

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

import config

TOL = 1e-5  # degrees, for the "grid is separable" test


def log(msg):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)


def expected_day_strings_and_time_units():
    with xr.open_dataset(config.ic_source_path, decode_times=False) as raw:
        units = raw[config.ocean_varnames["time"]].attrs.get("units")
    with xr.open_dataset(config.ic_source_path) as ic_full:
        time_values = ic_full[config.ocean_varnames["time"]].values
    days = [pd.Timestamp(t).strftime("%Y%m%d") for t in time_values]
    if not units or "since" not in units:
        units = f"days since {pd.Timestamp(time_values[0]):%Y-%m-%d} 00:00:00"
    return days, units


def separable_1d(da2d, along):
    """1-D values of a 2-D lon/lat array if it only varies along `along`."""
    if da2d is None or da2d.ndim != 2 or along not in da2d.dims:
        return None
    arr = np.asarray(da2d.transpose(*[d for d in da2d.dims]).values, dtype="f8")
    ax = da2d.dims.index(along)
    ref = arr[0, :] if ax == 1 else arr[:, 0]
    bcast = ref[None, :] if ax == 1 else ref[:, None]
    if not np.all(np.isfinite(arr)) or np.max(np.abs(arr - bcast)) > TOL:
        return None
    return ref


def gfdl_coords(ds):
    """Add/refresh the 1-D time/zl/ny/nx coordinates like 05 did."""
    for dim, axis, lonlat, units in (
        ("nx", "X", "xh", "degrees_east"),
        ("ny", "Y", "yh", "degrees_north"),
    ):
        if dim not in ds.dims:
            continue
        vals = separable_1d(ds.get(lonlat), dim)
        attrs = {"cartesian_axis": axis}
        if vals is not None:
            attrs["units"] = units
            note = f"lon/lat from {lonlat}"
        else:
            vals = np.arange(1, ds.sizes[dim] + 1, dtype="f8")
            note = "index values 1..N (grid not separable or no lon/lat)"
        ds = ds.assign_coords({dim: (dim, vals.astype("f8"), attrs)})
        log(f"  {dim}: {note}")
    if "zl" in ds.dims:
        ds["zl"].attrs.update({"cartesian_axis": "Z"})
        ds["zl"].attrs.setdefault("positive", "down")
    ds["time"].attrs["cartesian_axis"] = "T"
    return ds


def build_encoding(ds, time_units):
    enc = {}
    for name, var in ds.variables.items():
        if name == "time":
            enc[name] = {"dtype": "f8", "units": time_units,
                         "calendar": "gregorian", "_FillValue": None}
        elif name in ds.coords:
            enc[name] = {"_FillValue": None}
        elif var.dims and var.dims[0] == "time" and var.ndim >= 3:
            chunks = (1,) * (var.ndim - 2) + tuple(var.shape[-2:])
            enc[name] = {"chunksizes": chunks, "zlib": False, "_FillValue": 1e20}
    return enc


def combine(files, out_path, time_units):
    tmp_path = out_path.with_suffix(".nc.tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # "minimal"/"override": only variables with a time dim get stacked;
    # xh/yh/zl etc. come from the first file instead of being compared per
    # file or broadcast along time.
    with xr.open_mfdataset(
        files, combine="nested", concat_dim="time", chunks={"time": 1},
        data_vars="minimal", coords="minimal", compat="override",
    ) as ds:
        ds = gfdl_coords(ds)
        ds.to_netcdf(tmp_path, format="NETCDF4", unlimited_dims=["time"],
                     encoding=build_encoding(ds, time_units))
    verify(files, tmp_path)
    os.rename(tmp_path, out_path)
    log(f"Saved: {out_path}")


def verify(files, path):
    """Time unlimited + float64 + gregorian; spot-check first/middle/last day."""
    with nc.Dataset(path) as out:
        assert out.dimensions["time"].isunlimited(), "time is not unlimited"
        assert len(out.dimensions["time"]) == len(files), "time length mismatch"
        assert out["time"].dtype == np.dtype("f8"), "time is not float64"
        assert out["time"].calendar == "gregorian", "calendar is not gregorian"
        for dim in out.dimensions:
            assert dim in out.variables, f"no coordinate variable for {dim}"
    with xr.open_dataset(path) as out:
        for i in sorted({0, len(files) // 2, len(files) - 1}):
            with xr.open_dataset(files[i]) as day:
                for name in day.data_vars:
                    if "time" not in day[name].dims:
                        continue
                    a = day[name].isel(time=0).values
                    b = out[name].isel(time=i).values
                    assert np.array_equal(a, b, equal_nan=True), (
                        f"mismatch {name} at day index {i}")
    log(f"  checks passed for {path.name}")


def cross_check(paths):
    """Tracer and u/v files must share identical time/zl/ny/nx."""
    with nc.Dataset(paths[0]) as a, nc.Dataset(paths[1]) as b:
        for dim in ("time", "zl", "ny", "nx"):
            same = (dim in a.variables and dim in b.variables
                    and np.array_equal(a[dim][:], b[dim][:]))
            log(f"  {dim}: same in both files: {'yes' if same else 'NO'}")
            if not same:
                return False
    return True


def main():
    day_strs, time_units = expected_day_strings_and_time_units()
    log(f"Expecting {len(day_strs)} days, {day_strs[0]} to {day_strs[-1]}")

    tracer_files = [config.daily_stage_dir / f"ic_tracers_{d}.nc" for d in day_strs]
    uv_files = [config.daily_stage_dir / f"ic_uv_{d}.nc" for d in day_strs]

    outs = (config.ic_tracers_output_path, config.ic_uv_output_path)
    todo = [(files, out) for files, out in ((tracer_files, outs[0]), (uv_files, outs[1]))
            if not out.exists()]

    # A final file only appears (renamed from .tmp) after its checks pass, so
    # one that exists is complete: skip it. This makes a rerun after a killed
    # job pick up where it stopped.
    missing = [f for files, _ in todo for f in files if not f.exists()]
    if missing:
        print(f"ERROR: {len(missing)} per-day file(s) missing. First few:")
        for f in missing[:10]:
            print(f"  {f}")
        sys.exit(1)

    for out in outs:
        if out.exists():
            log(f"{out.name} already complete -- skipping.")
    for files, out in todo:
        stale = out.with_suffix(".nc.tmp")
        if stale.exists():
            log(f"Removing half-written {stale.name} from an earlier attempt.")
            stale.unlink()
        log(f"Combining into {out.name}...")
        combine(files, out, time_units)

    log("Cross-file grid check...")
    if not cross_check(outs):
        sys.exit("Grid mismatch between tracer and u/v files -- per-day files kept.")

    log("Cleaning up per-day files...")
    for f in tracer_files + uv_files:
        f.unlink(missing_ok=True)
    try:
        config.daily_stage_dir.rmdir()
    except OSError:
        log(f"{config.daily_stage_dir} not empty, left in place for inspection.")
    log("Done.")


if __name__ == "__main__":
    main()
