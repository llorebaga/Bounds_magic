#!/bin/bash
#SBATCH --job-name=magic-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Export the structures of one X-rank for the kernel (Python, memory-bound at n = 10).
# Set by rci.sh: R N D OUT; the array index is the X-rank a. Writes results/structs_r{R}_n{N}_d{D}_a{a}.txt.
set -eo pipefail
. "$MAGIC_DIR/cluster/env.sh"
a=$SLURM_ARRAY_TASK_ID
cd "$MAGIC_DIR/scripts"
{
    echo "EXPORT R=$R n=$N d=$D a=$a host=$(hostname)"
    R="$R" python export_structs.py "$N" "$D" "$a"
} > "$OUT/export_a$a.log.part" 2>&1
mv "$OUT/export_a$a.log.part" "$OUT/export_a$a.log"
