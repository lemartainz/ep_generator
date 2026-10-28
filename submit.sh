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

echo "Total events:    $N_EVENTS_TOTAL"
echo "Files to generate: $N_FILES"
echo "Jobs to submit:  $N_JOBS"
echo "Files per job:   $FILES_PER_JOB"
echo "Events per file: $EVENTS_PER_FILE"

sbatch --array=0-$(( N_JOBS - 1 ))%${CONCURRENCY} worker.sh $N_FILES $FILES_PER_JOB $EVENTS_PER_FILE
