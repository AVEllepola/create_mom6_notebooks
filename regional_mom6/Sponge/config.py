"""
Shared configuration for the Kimberly 10k Step 5b daily-IC pipeline.

Edit values here once -- 00_build_grid.py, 01_count_days.py,
02_process_one_day.py and 03_concatenate_days.py all import this file, so
they never drift out of sync with each other.
"""
import os
import sys
from pathlib import Path

# Dev checkouts of regional-mom6 / mom6_forge.
sys.path.insert(0, '/g/data/nm03/ae7501/regional-mom6')
sys.path.insert(0, '/g/data/nm03/ae7501/mom6_forge')

home = "/home/130/ae7501"
expt_name = "kimberly_1k_091126_sponge"

latitude_extent = [-21, -9.20]
longitude_extent = [114.8, 129.9]
date_range = ["2016-12-29 00:00:00", "2018-01-02 00:00:00"]

# Shared dir: holds the once-built grid (hgrid.nc / vcoord.nc), the per-day
# staging files, and the two final concatenated outputs.
input_dir = Path("/g/data/nm03/ae7501/glorys_data/sponge_data")

run_dir = Path(f"{home}/mom6_run_directories/2026_September/{expt_name}/")
fre_tools_dir = Path("/g/data/ik11/mom6_tools/tools/bin/")

# Raw multi-day IC source file.
glorys_path = Path("/g/data/nm03/ae7501/glorys_data/sponge_data_source")
ic_source_path = glorys_path / "ic_unprocessed_291216_020118.nc"

daily_stage_dir = input_dir / "ic_daily_tmp"
ic_tracers_output_path = input_dir / "ic_processed_291216_020118_tracers.nc"
ic_uv_output_path = input_dir / "ic_processed_291216_020118_uv.nc"

ocean_varnames = {
    "time": "time",
    "yh": "latitude",
    "xh": "longitude",
    "zl": "depth",
    "eta": "zos",
    "u": "uo",
    "v": "vo",
    "tracers": {"salt": "so", "temp": "thetao"},
}

# Shared kwargs for every rmom6.experiment(...) call. hgrid_type/vgrid_type
# are NOT here -- 00_build_grid.py and 02_process_one_day.py each set
# their own.
EXPERIMENT_KWARGS = dict(
    longitude_extent=longitude_extent,
    latitude_extent=latitude_extent,
    date_range=date_range,
    resolution=0.009,
    number_vertical_layers=100,
    layer_thickness_ratio=20,
    depth=5700,
    minimum_depth=5,
    mom_run_dir=run_dir,
    fre_tools_dir=fre_tools_dir,
    boundaries=["north", "south", "east", "west"],
    tidal_constituents=[],
)


def ensure_dirs():
    for path in (run_dir, glorys_path, input_dir, daily_stage_dir):
        os.makedirs(str(path), exist_ok=True)
