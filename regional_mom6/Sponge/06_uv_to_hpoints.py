#!/usr/bin/env python3
"""Convert a sponge UV file from staggered u/v points to cell-centre (h) points.

MOM6's ALE sponge reads u and v as arrays on the model's h-grid (nx x ny) and
averages neighbouring cells onto the u/v points itself, so the file must hold
    u(time, zl, ny, nx) and v(time, zl, ny, nx)
instead of your current
    u(time, zl, ny, nxp) and v(time, zl, nyp, nx).

Cell-centre values are the mean of the two faces that bound each cell:
    u_c[j, i] = 0.5 * (u[j, i] + u[j, i+1])   (west and east faces)
    v_c[j, i] = 0.5 * (v[j, i] + v[j+1, i])   (south and north faces)
If one face is missing (NaN or |value| > 1e20) the other face is used; if both
are missing the output is the fill value.

The output is a NEW file (the input is never modified) with:
  * time unlimited, float64, calendar gregorian, cartesian_axis T
  * 1-D coordinate variables time, zl, ny, nx (copied from the input; run
    05_fix_dimensions_like_gfdl.py on the input first so ny/nx exist)
  * u, v with dimensions (time, zl, ny, nx), same dtype as the input,
    no compression, one horizontal slab per HDF5 chunk
Parallel readers feed a single writer (concurrent writers to one HDF5 file
are not safe); memory use is small.

Usage:
    uv_to_hpoints.py IN_uv.nc OUT_uv_hgrid.nc [--workers 27] [--max-steps N]
"""
import argparse
import multiprocessing as mp
import os
import queue
import sys
import time as _time
from multiprocessing import shared_memory

import netCDF4 as nc
import numpy as np

BAD = 1e20  # |value| above this (or NaN) is treated as missing


def log(msg):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)


def avg_pair(a, b, fill):
    """Mean of a and b, using the valid one if the other is missing."""
    with np.errstate(invalid="ignore", over="ignore"):
        ba = ~np.isfinite(a) | (np.abs(a) > BAD)
        bb = ~np.isfinite(b) | (np.abs(b) > BAD)
        out = 0.5 * (a + b)
        out = np.where(ba & ~bb, b, out)
        out = np.where(bb & ~ba, a, out)
        out = np.where(ba & bb, fill, out)
    return out


def hslab(src, name, t, k, fill):
    """One (ny, nx) cell-centre slab computed from the staggered input."""
    if name == "u":
        x = src.variables["u"][t, k].astype("f8")  # (ny, nxp)
        return avg_pair(x[:, :-1], x[:, 1:], fill)
    x = src.variables["v"][t, k].astype("f8")  # (nyp, nx)
    return avg_pair(x[:-1, :], x[1:, :], fill)


def worker(src_path, tasks_q, free_q, done_q, shm_name, slab_bytes, fills):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        src = nc.Dataset(src_path, "r")
        src.set_auto_maskandscale(False)
        while True:
            task = tasks_q.get()
            if task is None:
                break
            name, t, k = task
            a = np.ascontiguousarray(hslab(src, name, t, k, fills[name]))
            buf = free_q.get()
            view = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf,
                               offset=buf * slab_bytes)
            view[...] = a
            done_q.put((name, t, k, buf, a.shape, a.dtype.str))
        src.close()
        shm.close()
    except Exception as e:
        done_q.put(("__error__", repr(e), 0, 0, 0, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=27)
    ap.add_argument("--max-steps", type=int, default=None,
                     help="only the first N time steps (timing test; NOT a full copy)")
    args = ap.parse_args()

    if os.path.exists(args.dst) and os.path.samefile(args.src, args.dst):
        sys.exit("REFUSING TO RUN: output and input are the same file")
    if os.path.realpath(args.src) == os.path.realpath(args.dst):
        sys.exit("REFUSING TO RUN: output and input are the same path")

    src = nc.Dataset(args.src, "r")
    src.set_auto_maskandscale(False)
    for need in ("u", "v", "time", "zl", "nx", "ny", "nxp", "nyp"):
        if need not in src.variables and need not in src.dimensions:
            sys.exit(f"input is missing '{need}' - run 05_fix_dimensions_like_gfdl.py "
                      f"first (it creates the 1-D nx/ny coordinate variables).")
    uvar, vvar = src["u"], src["v"]
    nt_src, nz = uvar.shape[0], uvar.shape[1]
    ny, nx = len(src.dimensions["ny"]), len(src.dimensions["nx"])
    assert uvar.shape[2:] == (ny, nx + 1), f"unexpected u shape {uvar.shape}"
    assert vvar.shape[2:] == (ny + 1, nx), f"unexpected v shape {vvar.shape}"
    nt = nt_src if args.max_steps is None else min(nt_src, args.max_steps)
    fills = {n: (float(src[n].getncattr("_FillValue")) if "_FillValue" in src[n].ncattrs()
                 else 9.969209968386869e36) for n in ("u", "v")}

    dst = nc.Dataset(args.dst, "w", format="NETCDF4")
    dst.set_auto_maskandscale(False)
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    dst.createDimension("time", None)
    dst.createDimension("zl", nz)
    dst.createDimension("ny", ny)
    dst.createDimension("nx", nx)

    # Coordinate variables.
    for name, ax in (("time", "T"), ("zl", "Z"), ("ny", "Y"), ("nx", "X")):
        v = src[name]
        out = dst.createVariable(name, "f8", (name,))
        out.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
        vals = v[:nt].astype("f8") if name == "time" else v[:].astype("f8")
        out[:] = vals
        out.setncattr("cartesian_axis", ax)
        if ax == "Z" and "positive" not in out.ncattrs():
            out.setncattr("positive", "down")
    dst["time"].setncattr("calendar", "gregorian")
    if not np.all(np.isfinite(dst["nx"][:])) or not np.all(np.isfinite(dst["ny"][:])):
        log("WARNING: nx/ny coordinate values contain NaN - run "
            "05_fix_dimensions_like_gfdl.py on the input first.")

    # Data variables on the h-grid.
    for name in ("u", "v"):
        v = src[name]
        out = dst.createVariable(name, v.dtype, ("time", "zl", "ny", "nx"),
                                  fill_value=fills[name], chunksizes=(1, 1, ny, nx))
        skip = {"_FillValue", "coordinates", "valid_min", "valid_max"}
        out.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k not in skip})
        out.setncattr("comment", "cell-centre values: mean of the two bounding faces "
                                  "of the staggered input (06_uv_to_hpoints.py)")
    log(f"output dims: time={nt}, zl={nz}, ny={ny}, nx={nx}; workers={args.workers}")

    tasks = [(name, t, k) for t in range(nt) for name in ("u", "v") for k in range(nz)]
    slab_bytes = ny * nx * 8
    src.close()

    nworkers = max(1, args.workers)
    nbuf = 2 * nworkers + 4
    shm = shared_memory.SharedMemory(create=True, size=nbuf * slab_bytes)
    ctx = mp.get_context("fork")
    tasks_q, free_q, done_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    for b in range(nbuf):
        free_q.put(b)
    for tk in tasks:
        tasks_q.put(tk)
    for _ in range(nworkers):
        tasks_q.put(None)
    procs = [ctx.Process(target=worker, args=(args.src, tasks_q, free_q, done_q,
                                               shm.name, slab_bytes, fills))
             for _ in range(nworkers)]
    for p in procs:
        p.start()

    t0 = _time.time()
    ntask = len(tasks)
    every = max(1, ntask // 100)
    try:
        for i in range(ntask):
            while True:
                try:
                    name, t, k, buf, shape, dt = done_q.get(timeout=60)
                    break
                except queue.Empty:
                    if all(not p.is_alive() for p in procs):
                        raise RuntimeError("all readers exited early")
            if name == "__error__":
                raise RuntimeError(f"reader failed: {t}")
            view = np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf,
                               offset=buf * slab_bytes)
            dst.variables[name][t, k] = view
            free_q.put(buf)
            if i % every == 0 or i == ntask - 1:
                el = _time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (ntask - i - 1) / rate / 60 if rate else 0
                log(f"{i + 1}/{ntask} slabs (time step {t + 1}/{nt}) "
                    f"elapsed {el / 60:.1f} min, ETA {eta:.1f} min")
    except Exception:
        for p in procs:
            p.terminate()
        raise
    finally:
        for p in procs:
            p.join(timeout=30)
        shm.close()
        shm.unlink()
        dst.close()

    # Verification against an independent recomputation.
    a, b = nc.Dataset(args.src), nc.Dataset(args.dst)
    a.set_auto_maskandscale(False)
    b.set_auto_maskandscale(False)
    assert b.dimensions["time"].isunlimited() and len(b.dimensions["time"]) == nt
    assert b["u"].shape == (nt, nz, ny, nx) and b["v"].shape == (nt, nz, ny, nx)
    for name in ("u", "v"):
        for t in sorted({0, nt // 2, nt - 1}):
            for k in sorted({0, nz // 2, nz - 1}):
                ref = hslab(a, name, t, k, fills[name])
                assert np.array_equal(ref, b[name][t, k]), f"mismatch {name} t={t} k={k}"
    log(f"OK: {args.dst} written: u, v on the h-grid (time unlimited, {nt} steps); "
        f"spot checks passed")


if __name__ == "__main__":
    main()
