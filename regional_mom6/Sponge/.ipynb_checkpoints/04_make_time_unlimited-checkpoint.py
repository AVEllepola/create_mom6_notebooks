#!/usr/bin/env python3
"""Copy a netCDF file, making the `time` dimension unlimited (record).

Parallel reader / single writer design:
* N reader processes open the input read-only and read slabs (one time
  step x one vertical level for 4-D variables) in parallel.
* Slabs travel to the writer through shared memory (no pickling of data).
* One writer (the main process) writes the output -- concurrent writers to
  a single netCDF-4/HDF5 file are not safe.

Raw values are copied with masking/scaling off, so fill values and packed
data are preserved exactly. No compression is applied. Memory use is small
(about (2*workers+4) slabs, under 1GB for the 1358x1677 grid).

Usage:
    04_make_time_unlimited.py IN.nc OUT.nc [--workers 13] [--max-steps N]

--max-steps copies only the first N time steps (for timing tests; the
output is then NOT a full copy).
"""
import argparse
import multiprocessing as mp
import queue
import sys
import time as _time
from multiprocessing import shared_memory

import netCDF4 as nc
import numpy as np


def log(msg):
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slab_of(v, t, k):
    return v[t] if k < 0 else v[t, k]


def worker(src_path, tasks_q, free_q, done_q, shm_name, slab_bytes):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        src = nc.Dataset(src_path, "r")
        src.set_auto_maskandscale(False)
        while True:
            task = tasks_q.get()
            if task is None:
                break
            name, t, k = task
            a = np.ascontiguousarray(slab_of(src.variables[name], t, k))
            buf = free_q.get()
            view = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf,
                               offset=buf * slab_bytes)
            view[...] = a
            done_q.put((name, t, k, buf, a.shape, a.dtype.str))
        src.close()
        shm.close()
    except Exception as e:
        # Report the error to the writer rather than hanging it.
        done_q.put(("__error__", repr(e), 0, 0, 0, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    src = nc.Dataset(args.src, "r")
    src.set_auto_maskandscale(False)
    if "time" not in src.dimensions:
        sys.exit("No dimension named 'time' in input file.")
    nt_src = len(src.dimensions["time"])
    nt = nt_src if args.max_steps is None else min(nt_src, args.max_steps)

    fmt = src.data_model
    is_nc4 = fmt.startswith("NETCDF4")
    dst = nc.Dataset(args.dst, "w", format=fmt)
    dst.set_auto_maskandscale(False)

    # Global attributes and dimensions (time -> unlimited).
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    for name, d in src.dimensions.items():
        dst.createDimension(name, None if name == "time" else len(d))

    # Variable definitions + attributes.
    for name, v in src.variables.items():
        kwargs = {}
        if "_FillValue" in v.ncattrs():
            kwargs["fill_value"] = v.getncattr("_FillValue")
        if is_nc4 and v.ndim == 4 and v.dimensions[0] == "time":
            kwargs["chunksizes"] = (1, 1) + tuple(v.shape[2:])  # 1 slab = 1 chunk
        out = dst.createVariable(name, v.dtype, v.dimensions, **kwargs)
        out.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})

    # Small/non-time variables and 1-D time-first variables copy directly;
    # everything else (time-first, >=2 dims) is copied slab by slab below.
    big = []
    for name, v in src.variables.items():
        if v.ndim >= 2 and v.dimensions[0] == "time":
            big.append(name)
        elif v.ndim >= 1 and v.dimensions[0] == "time":
            dst.variables[name][:nt] = v[:nt]
        else:
            dst.variables[name][...] = v[...]
    log(f"copied small variables; slab-copying {big} ({nt} time steps)")

    # Task list, time-major so a record is completed before the next starts.
    tasks, slab_bytes = [], 1
    for t in range(nt):
        for name in big:
            v = src.variables[name]
            if v.ndim == 4:
                tasks += [(name, t, k) for k in range(v.shape[1])]
                sb = int(np.prod(v.shape[2:])) * v.dtype.itemsize
            else:
                tasks.append((name, t, -1))
                sb = int(np.prod(v.shape[1:])) * v.dtype.itemsize
            slab_bytes = max(slab_bytes, sb)
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
    procs = [ctx.Process(target=worker,
                          args=(args.src, tasks_q, free_q, done_q, shm.name, slab_bytes))
             for _ in range(nworkers)]
    for p in procs:
        p.start()

    # Single writer: pull finished slabs off done_q and write them out.
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
            out = dst.variables[name]
            if k < 0:
                out[t] = view
            else:
                out[t, k] = view
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

    # Verify against the source.
    a, b = nc.Dataset(args.src), nc.Dataset(args.dst)
    a.set_auto_maskandscale(False)
    b.set_auto_maskandscale(False)
    assert b.dimensions["time"].isunlimited(), "time is not unlimited in output"
    assert len(b.dimensions["time"]) == nt, "time length mismatch"
    for name, v in a.variables.items():
        w = b.variables[name]
        if v.dimensions and v.dimensions[0] == "time":
            assert w.shape[0] == nt and w.shape[1:] == v.shape[1:], f"shape mismatch: {name}"
            for t in sorted({0, nt // 2, nt - 1}):
                if v.ndim == 4:
                    for k in sorted({0, v.shape[1] // 2, v.shape[1] - 1}):
                        assert np.array_equal(v[t, k], w[t, k]), f"mismatch {name} t={t} k={k}"
                else:
                    assert np.array_equal(v[t], w[t]), f"mismatch {name} t={t}"
        else:
            assert w.shape == v.shape and np.array_equal(v[...], w[...]), f"mismatch: {name}"
    log(f"OK: {args.dst} written, time is UNLIMITED ({nt} steps), spot checks passed")


if __name__ == "__main__":
    main()
