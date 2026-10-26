#!/bin/bash
#SBATCH --job-name=magic-mip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# One in-class MIP (two CCZ states, d = 2) per array task: task t -> (n, s) from the list. Needs Gurobi.
# Set by rci.sh: LIST OUT SOLVER_SECONDS.
set -eo pipefail
. "$MAGIC_DIR/cluster/env.sh"
[ -n "$GUROBI_MODULE" ] && module load "$GUROBI_MODULE"
t=$SLURM_ARRAY_TASK_ID
read -r n s < <(grep -v '^#' "$LIST" | grep . | sed -n "$((t + 1))p")
cd "$MAGIC_DIR/scripts"
THREADS=$SLURM_CPUS_PER_TASK python mip_scan.py ccz2 2 "$n" "$n" "$s" "$SOLVER_SECONDS" > "$OUT/task_$t.log.part" 2>&1
mv "$OUT/task_$t.log.part" "$OUT/task_$t.log"
