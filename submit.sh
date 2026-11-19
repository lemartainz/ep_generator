#!/bin/bash

# Submit from the repo so the jobs' SLURM_SUBMIT_DIR holds the macro and card.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

N_EVENTS_TOTAL="${1:-50000000}"   # e.g. 5e7
EVENTS_PER_FILE=5000
FILES_PER_JOB=100
CONCURRENCY="${2:-100}"

N_FILES=$(( N_EVENTS_TOTAL / EVENTS_PER_FILE ))        # 10000
N_JOBS=$(( (N_FILES + FILES_PER_JOB - 1) / FILES_PER_JOB ))  # 100 (ceiling division)

# Jobs inherit this shell's environment, so root has to be set up here.
if ! command -v root > /dev/null; then
    echo "ERROR: root not in PATH; load your ROOT/clas12 module before submitting." >&2
    exit 1
fi

# The card and output directory travel to the jobs through the environment:
#   INPUT=input_ps_iter0.txt OUTDIR=/volatile/.../LUND_files/ps_iter0 ./submit.sh 5e7
export INPUT="${INPUT:-input.txt}"
export OUTDIR="${OUTDIR:-/volatile/clas12/leomart/Simulations/LUND_files}"
if [ ! -f "$INPUT" ]; then
    echo "ERROR: card $INPUT not found in $(pwd)" >&2
    exit 1
fi
N_OLD=$(ls "$OUTDIR"/events_FD_*.dat 2>/dev/null | wc -l)
if [ "$N_OLD" -gt 0 ]; then
    echo "WARNING: $OUTDIR already holds $N_OLD events_FD_*.dat files; this run" \
         "overwrites them file by file. Set OUTDIR to a new directory per iteration." >&2
fi

echo "Card:            $INPUT"
echo "Output dir:      $OUTDIR"
echo "Total events:    $N_EVENTS_TOTAL"
echo "Files to generate: $N_FILES"
echo "Jobs to submit:  $N_JOBS"
echo "Files per job:   $FILES_PER_JOB"
echo "Events per file: $EVENTS_PER_FILE"

sbatch --array=0-$(( N_JOBS - 1 ))%${CONCURRENCY} worker.sh $N_FILES $FILES_PER_JOB $EVENTS_PER_FILE
