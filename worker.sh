#!/bin/bash
#SBATCH --partition=production
#SBATCH --job-name=generateLUND
#SBATCH --output=/farm_out/%u/%x-%A_%a.out
#SBATCH --error=/farm_out/%u/%x-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=leomart@jlab.org
#SBATCH --gres=disk:450G
#SBATCH --time=24:00:00

# Slurm runs a *copy* of this script from its spool directory, so
# BASH_SOURCE/dirname points there, not at the repo. SLURM_SUBMIT_DIR is the
# directory sbatch was called from (submit.sh cd's into the repo first).
cd "${SLURM_SUBMIT_DIR:?not running under slurm}" || exit 1
echo "Working dir: $(pwd) on $(hostname)"

MACRO="runEventGenerator.cpp"
INPUT="input.txt"
OUTDIR="${OUTDIR:-/volatile/clas12/leomart/Simulations/LUND_files}"

N_FILES_TOTAL="${1:?usage: worker.sh N_FILES_TOTAL FILES_PER_JOB [EVENTS_PER_FILE]}"
FILES_PER_JOB="${2:?usage: worker.sh N_FILES_TOTAL FILES_PER_JOB [EVENTS_PER_FILE]}"
EVENTS_PER_FILE="${3:-5000}"

JOB_IDX=$SLURM_ARRAY_TASK_ID
START_FILE=$(( JOB_IDX * FILES_PER_JOB + 1 ))
END_FILE=$(( START_FILE + FILES_PER_JOB - 1 ))

# Clamp to actual total in case the last job is a partial batch
[ "$END_FILE" -gt "$N_FILES_TOTAL" ] && END_FILE=$N_FILES_TOTAL

mkdir -p "$OUTDIR"

# Private scratch for this task. All array tasks running `runEventGenerator.cpp+`
# in the shared repo dir would compile runEventGenerator_cpp.so there at the
# same time and overwrite each other's library/dictionary mid-build, so each
# task compiles into its own ACLiC build dir instead.
SCRATCH="${TMPDIR:-/tmp}/generateLUND_${SLURM_ARRAY_JOB_ID}_${JOB_IDX}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
SET_BUILD_DIR="gSystem->SetBuildDir(\"${SCRATCH}\", kTRUE);"

# Per-task card: input.txt plus overrides (the reader keeps the last value of
# a key). The echo matters: input.txt has no trailing newline.
CARD="${SCRATCH}/input.txt"
{
    cat "$INPUT"
    echo
    echo "num_events: ${EVENTS_PER_FILE}"
    echo "write_lund: 1"
    echo "gen_plots: 0"
} > "$CARD"

# Compile once up front so a build error fails the task with a real exit code
# instead of being retried (and ignored) for every file.
root -l -b -q -e "$SET_BUILD_DIR" \
    -e "gSystem->Exit(gSystem->CompileMacro(\"${MACRO}\", \"k\") ? 0 : 1);"
if [ $? -ne 0 ]; then
    echo "ERROR: could not compile ${MACRO}" >&2
    exit 1
fi

echo "Job $JOB_IDX: files $START_FILE to $END_FILE, $EVENTS_PER_FILE events each"

N_FAILED=0
for i in $(seq "$START_FILE" "$END_FILE"); do
    OUT="${OUTDIR}/events_FD_${i}.dat"
    echo "  [$i] -> $OUT"
    rm -f "$OUT"
    root -l -b -q -e "$SET_BUILD_DIR" "${MACRO}+(\"${OUT}\",\"${CARD}\")"
    RC=$?
    if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
        echo "ERROR: file $i failed (root exit $RC, output missing or empty)" >&2
        N_FAILED=$(( N_FAILED + 1 ))
    fi
done

echo "Job $JOB_IDX done: $N_FAILED of $(( END_FILE - START_FILE + 1 )) files failed."
[ "$N_FAILED" -eq 0 ]
