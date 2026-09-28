#!/usr/bin/env python3
"""
List the days 02_process_one_day.py still has to do, as PBS array ranges.

A day counts as done only when BOTH ic_tracers_YYYYMMDD.nc and
ic_uv_YYYYMMDD.nc exist in the staging dir (02 writes them via .tmp +
rename, so an existing file is always complete).

Usage:
    python3 01_list_missing_days.py --count
        -> number of missing days, e.g. "37"
    python3 01_list_missing_days.py --total
        -> total number of days in the source file, e.g. "370"
    python3 01_list_missing_days.py --ranges --batch-size 10
        -> one "start-end" range per line, each at most batch-size days and
           covering only contiguous missing days, e.g.
               0-9
               10-12
               47-47
"""
import argparse

import pandas as pd
import xarray as xr

import config


def all_times():
    with xr.open_dataset(config.ic_source_path) as ic_full:
        return ic_full[config.ocean_varnames["time"]].values


def missing_day_indices(times):
    stage = config.daily_stage_dir
    missing = []
    for i, t in enumerate(times):
        d = pd.Timestamp(t).strftime("%Y%m%d")
        if not ((stage / f"ic_tracers_{d}.nc").exists()
                and (stage / f"ic_uv_{d}.nc").exists()):
            missing.append(i)
    return missing


def to_ranges(indices, batch_size):
    ranges, start, prev = [], None, None
    for i in indices:
        if start is None:
            start = prev = i
        elif i == prev + 1 and i - start + 1 <= batch_size:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    if start is not None:
        ranges.append((start, prev))
    return ranges


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--total", action="store_true")
    g.add_argument("--ranges", action="store_true")
    ap.add_argument("--batch-size", type=int, default=10)
    a = ap.parse_args()
    times = all_times()
    if a.total:
        print(len(times))
        return
    missing = missing_day_indices(times)
    if a.count:
        print(len(missing))
    else:
        for s, e in to_ranges(missing, a.batch_size):
            print(f"{s}-{e}")


if __name__ == "__main__":
    main()
