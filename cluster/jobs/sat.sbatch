#!/bin/bash
#SBATCH --job-name=magic-sat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
# One two-level SAT instance per array task (CaDiCaL is single-threaded). Set by rci.sh: LIST OUT SOLVER_SECONDS.
# A SAT answer is certified exactly by sat_sweep.py (verify_block.certify) and stored in the jsonl record.
set -eo pipefail
. "$MAGIC_DIR/cluster/env.sh"
t=$SLURM_ARRAY_TASK_ID
inst=$(grep -v '^#' "$LIST" | grep . | sed -n "$((t + 1))p")
cd "$MAGIC_DIR/scripts"
python sat_sweep.py "$OUT/task_$t.jsonl" "$SOLVER_SECONDS" "$inst" > "$OUT/task_$t.log.part" 2>&1
mv "$OUT/task_$t.log.part" "$OUT/task_$t.log"
