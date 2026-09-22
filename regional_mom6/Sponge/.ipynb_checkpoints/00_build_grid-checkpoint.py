#!/usr/bin/env python3
"""
Build the shared horizontal/vertical grid ONCE and write hgrid.nc / vcoord.nc
into input_dir.

Run this before submitting the per-day array job. Every per-day task in
process_one_day.py copies these two files into its own private scratch
directory and reuses them (hgrid_type="from_file", vgrid_type="from_file")
instead of regenerating the grid itself -- regeneration always overwrites,
so many parallel tasks all doing that against the same shared directory
would race each other writing the same hgrid.nc/vcoord.nc.

Per the regional_mom6 source, grid generation for the default
"even_spacing" / "hyperbolic_tangent" types is pure Python/NumPy -- no FRE
tools, no subprocess, no MPI -- so this should be fast (seconds, not hours).
"""
import time

import config
import regional_mom6 as rmom6

# Module-level, so this always prints as soon as the import above succeeds
# -- confirms which copy of regional_mom6 actually got loaded (the dev
# checkout config.py's sys.path.insert() points at, vs. whatever else
# might be on sys.path, e.g. one already in the conda environment). If
# this import had failed instead, you'd see a Python traceback
# (ModuleNotFoundError/ImportError) here and nothing below it would run.
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

    # __init__ already writes hgrid.nc eagerly for "even_spacing", but touch
    # both properties explicitly to be certain neither is left lazily
    # un-built (and un-written) by the time this script exits.
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

    # Sanity-check the source file now too, so a bad path is caught here
    # rather than 370 times over in parallel later.
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
