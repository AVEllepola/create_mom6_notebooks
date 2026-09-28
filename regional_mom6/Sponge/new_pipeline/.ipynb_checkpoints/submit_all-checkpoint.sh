#!/bin/bash
# Resumable driver for the Step 5b pipeline. Safe to run any number of times:
# every run looks at what is already on disk and only submits what is left.
#
#   1. 00_build_grid.pbs  -- build hgrid.nc / vcoord.nc (skipped if present)
#   2. process_days.pbs   -- PBS array jobs, only for days whose per-day files
#                            don't exist yet, at most MAX_ARRAYS_PER_ROUND
#                            arrays of BATCH_SIZE days per round
#   3. concatenate.pbs    -- 03_combine_days.py, once every day is done
#                            (itself resumable: skips a final file that
#                            already finished)
#
# Each round ends by submitting resubmit.pbs, which waits for that round's
# jobs to end (afterany -- success OR failure) and then runs this script
# again. So the pipeline keeps going by itself when:
#   * Gadi refuses more jobs (queue/array limit): the rest are submitted
#     next round, or after RETRY_DELAY_MIN if nothing could be submitted;
#   * some days fail or hit walltime: they are still "missing", so the next
#     round resubmits just those days.
# It stops by itself when both final files exist, or after MAX_NO_PROGRESS
# rounds in a row that submitted work but finished no new days (a day that
# keeps crashing), so it can never loop forever.
#
# If the chain ever breaks (e.g. resubmit.pbs itself was refused), just run
# ./submit_all.sh again by hand -- it picks up where things stopped.
#
# Needs the analysis3 conda environment loaded in THIS shell first:
#   module use /g/data/xp65/public/modules
#   module load conda/analysis3-26.09
#
# Usage: ./submit_all.sh            (FORCE=1 ./submit_all.sh to skip the
#                                    "jobs still running" safety check)
set -uo pipefail   # no -e on purpose: a refused qsub must not kill the round

# Must match config.py's input_dir.
INPUT_DIR="/g/data/nm03/ae7501/glorys_data/sponge_data"

BATCH_SIZE=10             # days per array job (Gadi accepted 10)
MAX_ARRAYS_PER_ROUND=20   # arrays submitted per round; lower it if Gadi
                          # still refuses jobs, raise it if it never does
RETRY_DELAY_MIN=30        # wait before retrying when nothing could be queued
MAX_NO_PROGRESS=3         # stop after this many rounds with no new days done

STATE_DIR=".pipeline_state"
mkdir -p "$STATE_DIR"
# Keep a running history of every round (resubmit.log is overwritten).
exec > >(tee -a "$STATE_DIR/history.log") 2>&1
echo
echo "===== submit_all.sh round at $(date) (job ${PBS_JOBID:-interactive}) ====="

# ---- 0. Don't start a second chain while one is still queued/running ------
if [[ -f "$STATE_DIR/active_jobs" && "${FORCE:-0}" != 1 ]]; then
    still=0
    while read -r id; do
        [[ -z "$id" || "$id" == "${PBS_JOBID:-none}" ]] && continue
        if qstat "$id" >/dev/null 2>&1; then
            echo "Still queued/running from an earlier round: $id"
            still=1
        fi
    done < "$STATE_DIR/active_jobs"
    if (( still )); then
        echo "Not submitting anything (would duplicate work). Wait for those"
        echo "jobs, or qdel them, or rerun with FORCE=1 ./submit_all.sh"
        exit 1
    fi
fi
: > "$STATE_DIR/active_jobs"
record() { echo "$1" >> "$STATE_DIR/active_jobs"; }

# A manual run means "try again": reset the no-progress counter.
[[ -z "${PBS_JOBID:-}" ]] && rm -f "$STATE_DIR/strikes"

# ---- 1. Already finished? -------------------------------------------------
read -r TR_OUT UV_OUT < <(python3 -c \
    "import config; print(config.ic_tracers_output_path, config.ic_uv_output_path)")
OUTS_MISSING=0
[[ -f "$TR_OUT" ]] || OUTS_MISSING=$((OUTS_MISSING + 1))
[[ -f "$UV_OUT" ]] || OUTS_MISSING=$((OUTS_MISSING + 1))
if (( OUTS_MISSING == 0 )); then
    echo "Both final files exist -- pipeline complete:"
    echo "  $TR_OUT"
    echo "  $UV_OUT"
    rm -f "$STATE_DIR/active_jobs" "$STATE_DIR/strikes" "$STATE_DIR/last_metric" \
          "$STATE_DIR/last_submitted"
    exit 0
fi

# ---- 2. Progress guard ----------------------------------------------------
if ! N_MISSING=$(python3 01_list_missing_days.py --count); then
    echo "01_list_missing_days.py failed -- fix that first."; exit 1
fi
N_TOTAL=$(python3 01_list_missing_days.py --total)
METRIC=$((N_MISSING + OUTS_MISSING))
PREV_METRIC=$(cat "$STATE_DIR/last_metric" 2>/dev/null || echo -1)
PREV_SUBMITTED=$(cat "$STATE_DIR/last_submitted" 2>/dev/null || echo 0)
STRIKES=$(cat "$STATE_DIR/strikes" 2>/dev/null || echo 0)
# Only count a strike if last round actually ran work and nothing improved
# (a round where the queue was full doesn't count against us).
if (( METRIC == PREV_METRIC && PREV_SUBMITTED > 0 )); then
    STRIKES=$((STRIKES + 1))
else
    STRIKES=0
fi
echo "$METRIC" > "$STATE_DIR/last_metric"
echo "$STRIKES" > "$STATE_DIR/strikes"
echo "$N_MISSING of $N_TOTAL days still to process; $OUTS_MISSING final file(s) missing."
if (( STRIKES >= MAX_NO_PROGRESS )); then
    echo "STOPPING: $STRIKES rounds in a row finished no new work."
    echo "These days keep failing -- check their process_days logs:"
    python3 01_list_missing_days.py --ranges --batch-size 1000000 | sed 's/^/  days /'
    echo "Fix the cause, then run ./submit_all.sh again."
    exit 1
fi

SUBMITTED=()

# ---- 3. Grid --------------------------------------------------------------
GRID_JOB=""
if [[ ! ( -f "$INPUT_DIR/hgrid.nc" && -f "$INPUT_DIR/vcoord.nc" ) ]]; then
    if GRID_JOB=$(qsub 00_build_grid.pbs 2>&1); then
        echo "Grid job: $GRID_JOB"; record "$GRID_JOB"; SUBMITTED+=("$GRID_JOB")
    else
        echo "qsub refused 00_build_grid.pbs: $GRID_JOB"; GRID_JOB=""
    fi
fi

# ---- 4. Days (or concatenation once all days are done) --------------------
if (( N_MISSING > 0 )) && [[ -f "$INPUT_DIR/hgrid.nc" || -n "$GRID_JOB" ]]; then
    DEP=()
    [[ -n "$GRID_JOB" ]] && DEP=(-W "depend=afterok:$GRID_JOB")
    mapfile -t RANGES < <(python3 01_list_missing_days.py --ranges --batch-size "$BATCH_SIZE")
    for r in "${RANGES[@]:0:$MAX_ARRAYS_PER_ROUND}"; do
        s=${r%-*}; e=${r#*-}
        # PBS needs >= 2 subjobs per array. Widen a lone day by one
        # neighbour; that neighbour is already done so 02 skips it in seconds.
        if (( s == e )); then
            if (( e + 1 < N_TOTAL )); then e=$((e + 1)); else s=$((s - 1)); fi
        fi
        if JOB=$(qsub ${DEP[@]+"${DEP[@]}"} -J "${s}-${e}" process_days.pbs 2>&1); then
            echo "Submitted days ${s}-${e}: $JOB"; record "$JOB"; SUBMITTED+=("$JOB")
        else
            echo "qsub refused days ${s}-${e}: $JOB"
            echo "Queue limit reached -- the rest go in the next round."
            break
        fi
    done
    if (( ${#RANGES[@]} > MAX_ARRAYS_PER_ROUND )); then
        echo "$(( ${#RANGES[@]} - MAX_ARRAYS_PER_ROUND )) more batch(es) held for later rounds."
    fi
elif (( N_MISSING == 0 )); then
    if JOB=$(qsub concatenate.pbs 2>&1); then
        echo "All days done -- concatenation job: $JOB"; record "$JOB"; SUBMITTED+=("$JOB")
    else
        echo "qsub refused concatenate.pbs: $JOB"
    fi
fi
echo "${#SUBMITTED[@]}" > "$STATE_DIR/last_submitted"

# ---- 5. Schedule the next round ------------------------------------------
if (( ${#SUBMITTED[@]} > 0 )); then
    DEPEND="afterany"
    for j in "${SUBMITTED[@]}"; do DEPEND="${DEPEND}:${j}"; done
    NEXT=$(qsub -W depend="$DEPEND" resubmit.pbs 2>&1)
    rc=$?
    WHY="when this round's ${#SUBMITTED[@]} job(s) end"
else
    WHEN=$(date -d "+${RETRY_DELAY_MIN} min" +%Y%m%d%H%M)
    NEXT=$(qsub -a "$WHEN" resubmit.pbs 2>&1)
    rc=$?
    WHY="in ${RETRY_DELAY_MIN} min (nothing could be queued this round)"
fi
if (( rc == 0 )); then
    record "$NEXT"
    echo "Next round ($NEXT) runs $WHY."
else
    echo "WARNING: could not queue the next round: $NEXT"
    echo "Run ./submit_all.sh again by hand once the queue has room."
fi

cat << EOF

Track progress with:
  qstat -u \$USER                            # all jobs
  qstat -t <a_batch_job_id>                  # subjobs within one batch
  python3 01_list_missing_days.py --count    # days still to do
  tail -f $STATE_DIR/history.log             # what each round did
EOF
