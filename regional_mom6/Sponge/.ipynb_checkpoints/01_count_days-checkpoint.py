#!/usr/bin/env python3
"""
Print the number of days (timesteps) in the source IC file.

Called by submit_all.sh to work out the day range for the process_days.pbs
array job, read straight from config.py so it can't drift from the source
file.

Needs the analysis3 conda environment loaded first (same one
process_days.pbs uses):
    module use /g/data/xp65/public/modules
    module load conda/analysis3-26.09
"""
import config
import xarray as xr


def main():
    with xr.open_dataset(config.ic_source_path) as ic_full:
        n_days = ic_full[config.ocean_varnames["time"]].sizes[
            config.ocean_varnames["time"]
        ]
    print(n_days)


if __name__ == "__main__":
    main()
