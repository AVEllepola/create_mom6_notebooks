#!/usr/bin/env python3
"""Give existing sponge/nudging files the dimension handling of GFDL's
nudging-file script, IN PLACE, without touching any data values.

GFDL's script writes files (time, depth, yh, xh) where:
  * time is UNLIMITED, float64, calendar "gregorian", cartesian_axis "T";
  * every dimension has a 1-D coordinate variable of the same name;
  * depth has cartesian_axis "Z" + positive "down"; xh/yh have "X"/"Y".
This script brings your files to the same state:

  1. checks `time` is the unlimited dimension (run 04_make_time_unlimited.py
     first if not);
  2. for every dimension, makes sure a 1-D coordinate variable of the same
     name exists and has cartesian_axis (T/Z/Y/X); depth gets positive="down".
     Missing coordinate variables are created; X/Y coordinates that are
     missing or all-NaN are filled with the real lon/lat (taken from your 2-D
     lon/lat variables when the grid is separable) or index values 1..N
     otherwise (a warning is printed);
  3. sets time:calendar = "gregorian" (xarray wrote "proleptic_gregorian");
  4. converts time from int64 to float64 (GFDL's dtype). The original int64
     variable is kept as `time_int64_backup`, with units/calendar removed.

Data variables (temp, salt, u, v, ...) are never modified; a fingerprint of
sample slabs is compared before and after to check.


Usage:
    05_fix_dimensions_like_gfdl.py FILE [FILE ...] [--dry-run]
        [--index-coords]      always use index values for X/Y (skip lon/lat lookup)
        [--keep-calendar]     do not change time:calendar
        [--keep-time-int64]   do not convert time to float64
        [--map NAME=AXIS ...] extra dimension-name -> axis (X/Y/Z/T) mappings
"""
import argparse
import hashlib
import sys

import netCDF4 as nc
import numpy as np

DEFAULT_MAP = {
    "time": "T",
    "zl": "Z", "zi": "Z", "depth": "Z", "z": "Z", "z_l": "Z", "z_i": "Z",
    "nx": "X", "nxp": "X", "xh": "X", "xq": "X", "lon": "X", "longitude": "X",
    "ny": "Y", "nyp": "Y", "yh": "Y", "yq": "Y", "lat": "Y", "latitude": "Y",
}
LON_RANGE, LAT_RANGE = (-360.0, 720.0), (-90.0, 90.0)
TOL = 1e-5  # degrees, for the "grid is separable" test


def fingerprint(path):
    """Hash a few raw data slabs of every time-first variable with >= 2 dims."""
    h = {}
    with nc.Dataset(path) as d:
        d.set_auto_maskandscale(False)
        for name, v in d.variables.items():
            if v.ndim >= 2 and v.dimensions[0] == "time" and v.shape[0] > 0:
                nt = v.shape[0]
                parts = []
                for t in sorted({0, nt - 1}):
                    x = v[t] if v.ndim == 2 else (v[t, v.shape[1] // 2])
                    parts.append(np.ascontiguousarray(x).tobytes())
                h[name] = hashlib.sha256(b"".join(parts)).hexdigest()[:16]
    return h


def derive_lonlat(d, dim, axis):
    """Return (values, source_name) for a 1-D lon/lat along `dim`, or (None, why)."""
    prefix = "x" if axis == "X" else "y"
    lo, hi = LON_RANGE if axis == "X" else LAT_RANGE
    why = "no 2-D " + prefix + "* variable uses this dimension"
    for name, v in d.variables.items():
        if v.ndim != 2 or dim not in v.dimensions or not name.lower().startswith(prefix):
            continue
        arr = v[:].astype("f8")
        along = v.dimensions.index(dim)
        ref = arr[0, :] if along == 1 else arr[:, 0]
        bcast = ref[None, :] if along == 1 else ref[:, None]
        if not np.all(np.isfinite(arr)):
            why = f"{name} contains NaN"
            continue
        if np.max(np.abs(arr - bcast)) > TOL:
            why = f"{name} is not separable (rotated/curvilinear grid)"
            continue
        if ref.min() < lo or ref.max() > hi:
            why = f"{name} values {ref.min():.3g}..{ref.max():.3g} outside {lo}..{hi}"
            continue
        return ref, name
    return None, why


def all_unset(vals):
    return not np.any(np.isfinite(vals))


def process(path, mapping, dry, index_coords, keep_cal, keep_int64):
    with nc.Dataset(path) as d:
        fmt = d.data_model
        if not fmt.startswith("NETCDF4"):
            print(f"\n{path}: {fmt} file - refusing to edit in place.")
            return False
        if "time" not in d.dimensions or not d.dimensions["time"].isunlimited():
            print(f"\n{path}: 'time' is not the unlimited dimension - run "
                  f"04_make_time_unlimited.py first.")
            return False
        before = fingerprint(path)

    d = nc.Dataset(path, "r" if dry else "a")
    d.set_auto_maskandscale(False)
    print(f"\n{path} [{d.data_model}]{' DRY RUN' if dry else ''}")

    for dim, dobj in d.dimensions.items():
        ax = mapping.get(dim)
        if ax is None:
            print(f"  {dim:<6}: axis unknown - skipped (use --map {dim}=X|Y|Z|T)")
            continue
        n = len(dobj)
        exists = dim in d.variables
        if exists and d.variables[dim].dimensions != (dim,):
            print(f"  {dim:<6}: variable '{dim}' is not 1-D over '{dim}' - skipped")
            continue

        # Coordinate values (only computed for X/Y that are missing or unset).
        new_vals, note = None, ""
        if ax in "XY":
            unset = (not exists) or all_unset(d.variables[dim][:].astype("f8"))
            if unset:
                if not index_coords:
                    new_vals, src = derive_lonlat(d, dim, ax)
                    note = f"lon/lat from {src}" if new_vals is not None else f"index values ({src})"
                if new_vals is None:
                    new_vals = np.arange(1, n + 1, dtype="f8")
                    note = note or "index values"
        elif not exists:
            new_vals = np.arange(1, n + 1, dtype="f8")
            note = "index values"

        action = []
        if not exists:
            action.append("create variable")
        if new_vals is not None:
            action.append(f"set values ({note})")
        action.append(f"cartesian_axis={ax}")
        print(f"  {dim:<6}: " + ", ".join(action))

        if dry:
            continue
        if not exists:
            d.createVariable(dim, "f8", (dim,))
        v = d.variables[dim]
        if new_vals is not None:
            v[:] = new_vals
            if "lon/lat" in note:
                v.setncattr("units", "degrees_east" if ax == "X" else "degrees_north")
        v.setncattr("cartesian_axis", ax)
        if ax == "Z" and "positive" not in v.ncattrs():
            v.setncattr("positive", "down")

    # Time: calendar + dtype.
    if "time" in d.variables:
        tv = d.variables["time"]
        cur = tv.getncattr("calendar") if "calendar" in tv.ncattrs() else None
        if not keep_cal and cur != "gregorian":
            print(f"  time  : calendar {cur!r} -> 'gregorian'")
            if not dry:
                tv.setncattr("calendar", "gregorian")
        if not keep_int64 and tv.dtype != np.dtype("f8"):
            if "time_int64_backup" in d.variables:
                print("  time  : 'time_int64_backup' already exists - not converting dtype")
            else:
                print(f"  time  : dtype {tv.dtype} -> float64 (original kept as time_int64_backup)")
                if not dry:
                    attrs = {k: tv.getncattr(k) for k in tv.ncattrs()}
                    vals = tv[:].astype("f8")
                    d.renameVariable("time", "time_int64_backup")
                    new = d.createVariable("time", "f8", ("time",))
                    new.setncatts(attrs)
                    new[:] = vals
                    bk = d.variables["time_int64_backup"]
                    for a in ("units", "calendar", "cartesian_axis", "axis"):
                        if a in bk.ncattrs():
                            bk.delncattr(a)
                    bk.setncattr("comment", "original int64 time, kept as backup; use 'time'")
    d.close()

    if dry:
        print("  (dry run, nothing written)")
        return True
    after = fingerprint(path)
    same = all(before[k] == after.get(k) for k in before)
    print("  data check: " + ("OK - sampled data slabs identical before/after" if same
                               else "MISMATCH - data changed!"))
    return same


def cross_check(files):
    """Compare grids across files (e.g. tracers vs uv) and check the staggering."""
    print("\n=== cross-file grid check ===")
    ds = [nc.Dataset(f) for f in files]
    for d in ds:
        d.set_auto_maskandscale(False)  # raw values, no masked arrays
    ok = True
    # Dimensions shared by name must have identical size and coordinates.
    names = set.intersection(*[set(d.dimensions) for d in ds])
    for dim in sorted(names):
        if not all(dim in d.variables for d in ds):
            continue
        sizes = {len(d.dimensions[dim]) for d in ds}
        if len(sizes) > 1:
            print(f"  {dim:<5}: SIZE MISMATCH {sizes}")
            ok = False
            continue
        if dim == "time":
            same = all(np.array_equal(ds[0][dim][:], d[dim][:]) for d in ds[1:])
        else:
            same = all(np.allclose(ds[0][dim][:], d[dim][:], equal_nan=True) for d in ds[1:])
        print(f"  {dim:<5}: same size and values in all files: {'yes' if same else 'NO'}")
        ok &= same
    # Staggering: nxp = nx + 1, nyp = ny + 1, faces half a cell from centres.
    for d, f in zip(ds, files):
        for cen, fac in (("nx", "nxp"), ("ny", "nyp")):
            if cen in d.dimensions and fac in d.dimensions:
                nc_, nf = len(d.dimensions[cen]), len(d.dimensions[fac])
                line = f"  {f.split('/')[-1]}: {fac}={nf} vs {cen}+1={nc_ + 1}"
                good = nf == nc_ + 1
                if fac in d.variables and cen in d.variables:
                    c, q = d[cen][:], d[fac][:]
                    if np.all(np.isfinite(c)) and np.all(np.isfinite(q)) and nc_ > 1:
                        half = 0.5 * np.median(np.diff(c))
                        off = float(q[0] - c[0])
                        line += f", first face offset {off:.5g} (half-cell {half:.5g})"
                        good &= bool(np.isclose(abs(off), abs(half), rtol=0.05))
                print(line + (" OK" if good else " CHECK THIS"))
                ok &= good
    for d in ds:
        d.close()
    print("  grid check: " + ("PASSED" if ok else "PROBLEMS FOUND - see above"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--index-coords", action="store_true")
    ap.add_argument("--keep-calendar", action="store_true")
    ap.add_argument("--keep-time-int64", action="store_true")
    ap.add_argument("--map", action="append", default=[], metavar="NAME=AXIS")
    a = ap.parse_args()
    mapping = dict(DEFAULT_MAP)
    for m in a.map:
        k, _, v = m.partition("=")
        if len(v) != 1 or v.upper() not in "XYZT":
            sys.exit(f"bad --map value: {m}")
        mapping[k] = v.upper()
    ok = all([process(f, mapping, a.dry_run, a.index_coords, a.keep_calendar,
                       a.keep_time_int64) for f in a.files])
    if ok and not a.dry_run:
        ok = cross_check(a.files) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
