#!/bin/bash
# Submits the full three-stage Step 5b pipeline with PBS job dependencies so
# each stage only starts once the previous one has finished successfully:
#
#   1. 00_build_grid.pbs   -- build hgrid.nc / vcoord.nc once (skipped
#                              automatically if both already exist)
#   2. process_days.pbs, BATCHED -- one PBS array job per 10-day batch
#                              (Gadi's PBS server rejects a single
#                              array bigger than that: "qsub: Array
#                              job exceeds server or queue size
#                              limit"), submitted in a loop below
#                              until every day is covered.
#   3. concatenate.pbs     -- merge every per-day file into the two
#                              final outputs, once every batch from
#                              stage 2 has finished successfully.
#
# Needs the analysis3 conda environment loaded in THIS shell first (not
# just inside the .pbs scripts), since 01_count_days.py below reads the day
# count directly:
#   module use /g/data/xp65/public/modules
#   module load conda/analysis3-26.09

#   module use /g/data/vk83/modules
#   module load rmom6
#
# Usage: ./submit_all.sh
set -e

# Must match config.py's input_dir -- kept as a literal here since this is
# bash, not Python, so it can't just import config.py directly.
INPUT_DIR="/g/data/nm03/ae7501/glorys_data/sponge_data"

# Max subjobs per array submission. Gadi's PBS server rejected 370 in one
# go; 10 is the size confirmed to work. Lower this further if you ever see
# "Array job exceeds server or queue size limit" again.
BATCH_SIZE=10

GRID_JOB=""
if [[ -f "$INPUT_DIR/hgrid.nc" && -f "$INPUT_DIR/vcoord.nc" ]]; then
    echo "Found existing $INPUT_DIR/hgrid.nc and vcoord.nc -- skipping 00_build_grid.pbs."
    echo "(Delete both, or edit this script, if you actually want to rebuild them.)"
else
    GRID_JOB=$(qsub 00_build_grid.pbs)
    echo "Grid job: $GRID_JOB"
fi

N=$(python3 01_count_days.py)
echo "Found $N days to process, in batches of $BATCH_SIZE"

DAY_JOB_IDS=()
start=0
while [ "$start" -lt "$N" ]; do
    end=$((start + BATCH_SIZE - 1))
    if [ "$end" -ge "$N" ]; then
        end=$((N - 1))
    fi
    if [[ -n "$GRID_JOB" ]]; then
        JOB=$(qsub -J "${start}-${end}" -W depend=afterok:"$GRID_JOB" process_days.pbs)
    else
        JOB=$(qsub -J "${start}-${end}" process_days.pbs)
    fi
    echo "Submitted days ${start}-${end}: $JOB"
    DAY_JOB_IDS+=("$JOB")
    start=$((end + 1))
done
echo "Submitted ${#DAY_JOB_IDS[@]} batch array jobs covering days 0-$((N - 1))"

# Concatenation waits on every batch's array finishing successfully.
# afterokarray accepts multiple job ids colon-separated after one prefix.
DEPEND="afterokarray"
for JOB in "${DAY_JOB_IDS[@]}"; do
    DEPEND="${DEPEND}:${JOB}"
done
CONCAT_JOB=$(qsub -W depend="$DEPEND" concatenate.pbs)
echo "Concatenation job: $CONCAT_JOB"

cat << EOF

Track progress with:
  qstat -u \$USER              # overall status of every job/batch
  qstat -t <a_batch_job_id>    # per-day subjob status within one batch
  tail -f build_grid.log
  tail -f process_days_<N>.log # one log per day, e.g. process_days_47.log
  tail -f concat_statefiles.log
EOF
