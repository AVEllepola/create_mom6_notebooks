#!/usr/bin/env python3
"""
Build the shared horizontal/vertical grid once, writing hgrid.nc / vcoord.nc
into input_dir. Run this before submitting the per-day array job.

Every per-day task in 02_process_one_day.py copies these two files into its
own scratch dir instead of regenerating them -- regenerating would make
parallel tasks race to overwrite the same shared files.
"""
import time

import config
import regional_mom6 as rmom6

# Confirms which regional_mom6 copy got loaded (dev checkout vs conda env).
print(f"Using regional_mom6 from: {rmom6.__file__}")


def main():
    config.ensure_dirs()

    print(f"Building grid in {config.input_dir} ...")
    t0 = time.time()
    expt = rmom6.experiment(
        mom_input_dir=config.input_dir,
        hgrid_type="even_spacing",
        vgrid_type="hyperbolic_tangent",
        **config.EXPERIMENT_KWARGS,
    )

    # Force both grids to build now, so neither is left un-written lazily.
    _ = expt.hgrid
    _ = expt.vgrid
    print(f"Grid built in {time.time() - t0:.1f}s")

    for fname in ("hgrid.nc", "vcoord.nc"):
        fpath = config.input_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected {fpath} to exist after building the grid, but it "
                "doesn't -- check this regional_mom6 version's grid-writing "
                "behaviour before running the per-day array job."
            )
        print(f"  wrote {fpath} ({fpath.stat().st_size} bytes)")

    # Sanity-check the source file here once, rather than 370 times in parallel.
    import xarray as xr
    with xr.open_dataset(config.ic_source_path) as ic_full:
        n_days = ic_full[config.ocean_varnames["time"]].sizes[
            config.ocean_varnames["time"]
        ]
    print(f"Source file has {n_days} timesteps: {config.ic_source_path}")
    print(f"When submitting process_days.pbs, use -J 0-{n_days - 1}%<concurrency>")

    print("Done. You can now submit the per-day array job.")


if __name__ == "__main__":
    main()
